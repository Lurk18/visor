"""
Servicio web UACANet-v2 (segmentacion de polipos).

Reutiliza viewer_uacanet.py (debe estar en la misma carpeta) para no duplicar
el modelo ni el preprocesado. El navegador captura el video y envia frames
JPEG por WebSocket; el servidor devuelve la probabilidad de segmentacion como
PNG de 8 bits. El umbral y el dibujo se aplican en el cliente (slider instantaneo).

    CHECKPOINT=best_model.pth uvicorn server:app --host 0.0.0.0 --port 8000

Protocolo WebSocket /ws:
    cliente -> servidor : bytes JPEG del frame
    servidor -> cliente : 4 bytes (uint32 big-endian, id de secuencia) + PNG (prob 0-255)
                          El id viene en los primeros 4 bytes del frame recibido.
Para ello el cliente antepone 4 bytes de id a cada JPEG.
"""
import asyncio
import base64
import csv
from datetime import datetime
import io
import os
import re
import json
import secrets
import shutil
import tempfile
import threading
import time
from urllib.parse import quote

import cv2
import httpx
import numpy as np
import torch
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from viewer_uacanet import load_model, segment_frame



def _load_dotenv(path):
    """Carga un archivo .env (KEY=VALOR) sin dependencias extra. Util en Windows, donde
    la sintaxis `VAR=valor comando` de bash no existe. Las variables ya definidas ganan."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ.setdefault(k, v)


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def clean_supabase_url(u: str) -> str:
    """Deja la URL como https://<ref>.supabase.co (sin comillas, espacios, /rest/v1, etc.)."""
    u = (u or "").strip().strip("\"'").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    u = re.sub(r"/(rest|auth|storage|realtime|functions)/v1.*$", "", u, flags=re.I)
    return u.rstrip("/")


def clean_key(k: str) -> str:
    return re.sub(r"\s+", "", (k or "").strip().strip("\"'"))


def jwt_role(key: str):
    """Rol declarado dentro de una clave JWT clasica (anon / service_role); None si no es JWT."""
    try:
        payload = key.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("role")
    except Exception:
        return None


CHECKPOINT = os.environ.get("CHECKPOINT", "best_model.pth")
SIZE = int(os.environ.get("MODEL_SIZE", "352"))
MAX_BYTES = 4 * 1024 * 1024  # limite por frame
# Raiz del almacenamiento (en el cluster: COLONOSCOPIAS_DIR=/home/Data/Colonoscopias)
REGISTROS_DIR = os.environ.get("COLONOSCOPIAS_DIR") or os.environ.get("REGISTROS_DIR", "./registros")
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_MB", "2048")) * 1024 * 1024
PID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")   # ID que da el hospital
SESION_RE = re.compile(r"^colonoscopia_\d{4}_\d{3,}$")        # colonoscopia_<año>_<consecutivo global del año>
SUBDIRS = ("video", "predictions/masks", "predictions/frames", "annotations/manual_masks",
           "annotations/frames", "transcriptions", "metadata/miniaturas")
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}
history_lock = threading.Lock()

# ---------- autenticacion con Supabase ----------
SUPABASE_URL = clean_supabase_url(os.environ.get("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = clean_key(os.environ.get("SUPABASE_ANON_KEY", ""))  # clave publica (anon); NUNCA la service_role
if jwt_role(SUPABASE_ANON_KEY) == "service_role":
    print("ERROR DE SEGURIDAD: SUPABASE_ANON_KEY contiene la clave service_role. Se ignora; usa la clave anon/publishable.")
    SUPABASE_ANON_KEY = ""
_token_cache = {}  # token -> (expira, usuario); evita llamar a Supabase en cada request


async def verify_token(token: str):
    """Valida el JWT preguntandole a Supabase Auth (sirve con cualquier tipo de firma)."""
    if not token or not SUPABASE_URL:
        return None
    hit = _token_cache.get(token)
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{SUPABASE_URL}/auth/v1/user",
                            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY})
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    user = r.json()
    now = time.time()
    for k in [k for k, v in _token_cache.items() if v[0] < now]:
        del _token_cache[k]
    _token_cache[token] = (now + 60, user)
    return user


async def current_user(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ").strip()
    user = await verify_token(token)
    if not user:
        raise HTTPException(401, "No autenticado")
    return {**user, "_token": token}


app = FastAPI(title="UACANet-v2 Polyp Segmentation")
router = APIRouter()   # las rutas viven aqui y se montan en la raiz y bajo BASE_PATHS (ver el final del archivo)
state = {}
gpu_lock = threading.Lock()  # una inferencia a la vez; el modelo no es thread-safe en la practica


def check_supabase():
    """Comprueba al arrancar que Supabase es alcanzable desde este equipo y lo deja en consola."""
    state["supabase"] = "sin configurar"
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("AVISO: faltan SUPABASE_URL / SUPABASE_ANON_KEY; nadie podra iniciar sesion.")
        print("       Crea un archivo .env junto a server.py con esas dos variables (ver README).")
        return
    print(f"Supabase URL: {SUPABASE_URL}")
    try:
        r = httpx.get(f"{SUPABASE_URL}/auth/v1/settings", headers={"apikey": SUPABASE_ANON_KEY}, timeout=8)
    except httpx.HTTPError as e:
        state["supabase"] = f"inalcanzable: {type(e).__name__}"
        print(f"ERROR: no se pudo conectar con Supabase ({type(e).__name__}: {e}).")
        print("       Revisa que SUPABASE_URL sea correcta (Project Settings -> API), que el proyecto no este")
        print("       pausado (dashboard de Supabase) y que haya internet / el firewall no lo bloquee.")
        return
    if r.status_code == 200:
        state["supabase"] = "ok"
        print("Supabase: conexion OK.")
    elif r.status_code in (401, 403):
        state["supabase"] = "clave invalida"
        print(f"ERROR: Supabase respondio {r.status_code}; la SUPABASE_ANON_KEY parece incorrecta.")
    else:
        state["supabase"] = f"http {r.status_code}"
        print(f"AVISO: Supabase respondio {r.status_code} en /auth/v1/settings.")


@app.on_event("startup")
def _load():
    check_supabase()
    state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["model"] = load_model(CHECKPOINT, state["device"])


def infer_png(jpeg: bytes) -> bytes:
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG invalido")
    with gpu_lock:
        prob, _ = segment_frame(state["model"], frame, state["device"], size=SIZE)
    ok, png = cv2.imencode(".png", (prob * 255).astype(np.uint8))
    return png.tobytes()


@router.get("/api/config")
def public_config():
    """Datos publicos que necesita el navegador para iniciar sesion."""
    return {"supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}


@router.get("/health")
def health():
    return {"status": "ok", "device": str(state.get("device")), "model_loaded": "model" in state,
            "supabase": state.get("supabase")}


@router.post("/api/segment")
async def segment_image(file: UploadFile = File(...), user: dict = Depends(current_user)):
    """Una sola imagen -> PNG con la probabilidad (0-255)."""
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Imagen demasiado grande")
    try:
        png = await asyncio.to_thread(infer_png, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(png, media_type="image/png")


@router.websocket("/ws")
async def ws_segment(ws: WebSocket):
    await ws.accept()
    try:  # el primer mensaje debe ser {"token": "<jwt>"}
        first = await asyncio.wait_for(ws.receive_text(), timeout=10)
        user = await verify_token(json.loads(first).get("token", ""))
    except Exception:
        user = None
    if not user:
        try:
            await ws.close(code=4401)
        except Exception:
            pass
        return
    try:
        while True:
            msg = await ws.receive_bytes()
            if len(msg) < 5 or len(msg) > MAX_BYTES:
                continue
            seq, jpeg = msg[:4], msg[4:]
            try:
                png = await asyncio.to_thread(infer_png, jpeg)
            except ValueError:
                continue
            await ws.send_bytes(seq + png)
    except WebSocketDisconnect:
        pass


# ---------- registros: <raiz>/<usuario_id>/paciente_<ID>/colonoscopia_<año>_<NNN>/ ----------
def user_dir(user: dict) -> str:
    uid = re.sub(r"[^A-Za-z0-9_-]", "_", user.get("id", ""))[:64] or "anon"
    return os.path.join(REGISTROS_DIR, uid)


def patient_dir(user: dict, paciente: str) -> str:
    if not PID_RE.match(paciente or ""):
        raise HTTPException(400, "ID de paciente invalido (letras, numeros, . - _; maximo 40)")
    return os.path.join(user_dir(user), "paciente_" + paciente)


def session_dir(user: dict, paciente: str, sesion: str) -> str:
    if not SESION_RE.match(sesion or ""):
        raise HTTPException(400, "Colonoscopia invalida")
    return os.path.join(patient_dir(user, paciente), sesion)


def P(d, *parts):
    return os.path.join(d, *parts)


def jread(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def jwrite(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def reserve_code():
    """Consecutivo GLOBAL por año (todos los usuarios y pacientes): colonoscopia_2026_001, _002, ...
    Se reserva con un archivo creado en modo exclusivo en <raiz>/.indice, asi no se repite ni con varios procesos."""
    year = time.strftime("%Y")
    idx = os.path.join(REGISTROS_DIR, ".indice")
    os.makedirs(idx, exist_ok=True)
    n = 1 + max([int(m.group(1)) for f in os.listdir(idx) if (m := re.match(rf"^{year}_(\d+)$", f))], default=0)
    while True:
        path = os.path.join(idx, f"{year}_{n:03d}")
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return f"colonoscopia_{year}_{n:03d}", path
        except FileExistsError:
            n += 1


def link_notes(d):
    """Cada nota (descripcion o transcripcion) lista los hallazgos que hay en su mismo frame."""
    frames = {}
    for p in jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", []):
        if not p.get("descartado"):
            frames.setdefault(p["frame"], []).append("automatico")
    for c in jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", []):
        if not c.get("descartado"):
            frames.setdefault(c["frame"], []).append("manual")
    for path, key in ((P(d, "annotations", "descriptions.json"), "descripciones"),
                      (P(d, "transcriptions", "transcript.json"), "transcripciones")):
        data = jread(path, {})
        changed = False
        for n in data.get(key, []):
            t = frames.get(n["frame"], [])
            if n.get("hallazgos_en_frame") != t:
                n["hallazgos_en_frame"] = t
                changed = True
        if changed:
            jwrite(path, data)


# ---------- CSV: notas y hallazgos ----------
# notas.csv queda en la carpeta de la colonoscopia (se reescribe con cada nota y al finalizar) y tambien se puede descargar desde el visor.
# UTF-8 con BOM para que Excel muestre bien los acentos. Si tu Excel lo abre todo en una columna, arranca con CSV_SEP=";".
CSV_SEP = os.environ.get("CSV_SEP", ",")[:1] or ","


def _csv_safe(v):
    """Evita que Excel interprete como formula un texto dictado que empiece por = + - @."""
    v = "" if v is None else str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v


def notes_csv_text(d):
    rows = []
    for path, key, origen in ((P(d, "annotations", "descriptions.json"), "descripciones", "texto"),
                              (P(d, "transcriptions", "transcript.json"), "transcripciones", "voz")):
        rows += [(n.get("frame", 0), n.get("creado", ""), n, origen) for n in jread(path, {}).get(key, [])]
    rows.sort(key=lambda r: (r[0], r[1]))
    out = io.StringIO()
    w = csv.writer(out, delimiter=CSV_SEP, lineterminator="\r\n")
    w.writerow(["frame", "tiempo", "texto", "origen", "hallazgo", "hallazgos_en_frame", "creado"])
    for frame, _, n, origen in rows:
        w.writerow([frame, n.get("tiempo", ""), _csv_safe(n.get("texto", "")), origen, n.get("hallazgo", ""),
                    "|".join(n.get("hallazgos_en_frame", [])), n.get("creado", "")])
    return out.getvalue()


def findings_csv_text(d):
    rows = [["frame", "tiempo", "tipo", "forma", "x", "y", "ancho_caja", "alto_caja", "mascara", "frame_img"]]
    for p in jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", []):
        if p.get("descartado"):
            continue
        b = p.get("box") or ["", "", "", ""]
        rows.append([p["frame"], p["tiempo"], "automatico", "", *b, p.get("mascara") or "", p.get("frame_img") or ""])
    for c in jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", []):
        if c.get("descartado"):
            continue
        b = c.get("box") or ["", "", "", ""]
        rows.append([c["frame"], c["tiempo"], "manual", c.get("forma", ""), *b, c.get("mascara") or "", c.get("frame_img") or ""])
    body = io.StringIO()
    csv.writer(body).writerows(rows)
    return body.getvalue()


def write_notes_csv(d):
    with open(P(d, "notas.csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write(notes_csv_text(d))


def refresh_notes_csv(d):          # al llegar un hallazgo nuevo cambia la columna hallazgos_en_frame de las notas ya guardadas
    if os.path.exists(P(d, "notas.csv")):
        write_notes_csv(d)


# ---------- metadatos de los estudios en Supabase (tabla `estudios`, ver supabase_estudios.sql) ----------
SUPABASE_TABLA = os.environ.get("SUPABASE_TABLA_ESTUDIOS", "estudios")
SUPABASE_ACTIVO = os.environ.get("SUPABASE_SYNC_ESTUDIOS", "1") != "0" and bool(SUPABASE_URL and SUPABASE_ANON_KEY)
SYNC_DEBOUNCE = float(os.environ.get("SYNC_DEBOUNCE", "2"))   # segundos: junta rafagas de eventos en un solo envio
_sync_pending = set()


def video_completo(meta):
    v = meta.get("video")
    return bool(v) and (not v.get("total") or v.get("bytes", 0) >= v["total"])


def _iso_local(texto):
    try:
        return datetime.fromisoformat(texto).astimezone().isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def study_counts(d):
    vivos = lambda xs: len([x for x in xs if not x.get("descartado")])
    return (vivos(jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", [])),
            vivos(jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", [])),
            len(jread(P(d, "annotations", "descriptions.json"), {}).get("descripciones", []))
            + len(jread(P(d, "transcriptions", "transcript.json"), {}).get("transcripciones", [])))


def study_row(d, user, paciente, sesion):
    meta = jread(P(d, "metadata", "colonoscopy.json"), {})
    video = meta.get("video") or {}
    n_pred, n_caj, n_notas = study_counts(d)
    return {"codigo": sesion, "usuario_id": user["id"], "paciente_id": paciente,
            "ruta": f"{os.path.basename(user_dir(user))}/paciente_{paciente}/{sesion}",
            "fuente": meta.get("fuente"), "creado": _iso_local(meta.get("creado", "")),
            "actualizado": datetime.now().astimezone().isoformat(timespec="seconds"),
            "video_bytes": video.get("bytes"), "video_total": video.get("total"),
            "n_predicciones": n_pred, "n_anotaciones": n_caj, "n_notas": n_notas,
            "ancho": meta.get("ancho"), "alto": meta.get("alto")}


def _push_study(user, d, paciente, sesion):
    try:
        time.sleep(SYNC_DEBOUNCE)
        with history_lock:
            _sync_pending.discard(d)
            row = study_row(d, user, paciente, sesion)
        estado = {"estado": "ok", "intento": now_iso()}
        try:
            r = httpx.post(f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLA}?on_conflict=codigo", json=row, timeout=10,
                           headers={"apikey": SUPABASE_ANON_KEY, "Authorization": "Bearer " + user["_token"],
                                    "Content-Type": "application/json",
                                    "Prefer": "resolution=merge-duplicates,return=minimal"})
            if r.status_code not in (200, 201, 204):
                estado = {"estado": "error", "intento": now_iso(), "http": r.status_code, "detalle": r.text[:200]}
        except httpx.HTTPError as e:
            estado = {"estado": "pendiente", "intento": now_iso(), "detalle": str(e)[:200]}
        with history_lock:
            meta = jread(P(d, "metadata", "colonoscopy.json"), {})
            meta["supabase"] = estado
            jwrite(P(d, "metadata", "colonoscopy.json"), meta)
    finally:
        with history_lock:
            _sync_pending.discard(d)


def sync_study(user, d, paciente, sesion):
    """Programa, sin bloquear la peticion, el envio de los metadatos del estudio a Supabase.
    Si Supabase no responde, queda como 'pendiente' y se reintenta con el siguiente evento del estudio."""
    if not SUPABASE_ACTIVO:
        return
    with history_lock:
        if d in _sync_pending:
            return
        _sync_pending.add(d)
    threading.Thread(target=_push_study, args=(user, d, paciente, sesion), daemon=True).start()


@router.post("/api/sessions")
def create_session(paciente: str = Form(...), fuente: str = Form(""), user: dict = Depends(current_user)):
    """Abre una colonoscopia nueva del paciente y devuelve su codigo (colonoscopia_<año>_<NNN>)."""
    base = patient_dir(user, paciente)
    with history_lock:
        name, idxfile = reserve_code()
        d = os.path.join(base, name)
        for sd in SUBDIRS:
            os.makedirs(P(d, *sd.split("/")))
        with open(idxfile, "w", encoding="utf-8") as f:
            f.write(f"{os.path.basename(user_dir(user))}/paciente_{paciente}/{name}\n")
        jwrite(P(d, "metadata", "colonoscopy.json"),
               {"estudio": name, "usuario_id": os.path.basename(user_dir(user)), "paciente_id": paciente,
                "fuente": fuente[:120], "creado": now_iso(), "video": None, "ancho": None, "alto": None})
        jwrite(P(d, "predictions", "metadata.json"),
               {"modelo": os.path.basename(CHECKPOINT), "criterio": "un registro al iniciar cada deteccion",
                "ancho": None, "alto": None, "predicciones": []})
        jwrite(P(d, "annotations", "bounding_boxes.json"), {"ancho": None, "alto": None, "cajas": []})
        jwrite(P(d, "annotations", "descriptions.json"), {"descripciones": []})
        jwrite(P(d, "transcriptions", "transcript.json"),
               {"motor": "Web Speech API (navegador)", "idioma": "es-419", "transcripciones": []})
    sync_study(user, d, paciente, name)
    return {"sesion": name}


@router.get("/api/sessions")
def list_sessions(paciente: str, user: dict = Depends(current_user)):
    base = patient_dir(user, paciente)
    out = []
    if os.path.isdir(base):
        with history_lock:
            for n in sorted(os.listdir(base), reverse=True):
                d = os.path.join(base, n)
                if not (SESION_RE.match(n) and os.path.isdir(d)):
                    continue
                meta = jread(P(d, "metadata", "colonoscopy.json"), {})
                out.append({"sesion": n, "creado": meta.get("creado", ""), "fuente": meta.get("fuente", ""),
                            "hallazgos": sum(study_counts(d)[:2]),
                            "tiene_video": video_completo(meta)})
    return out


@router.put("/api/sessions/video")
async def upload_video(request: Request, paciente: str, sesion: str, ext: str = ".mp4",
                       offset: int = 0, total: int = 0, user: dict = Depends(current_user)):
    """Escribe el cuerpo en video/original.<ext> a partir de `offset` (subida por trozos, reanudable e idempotente).
    offset=0 empieza el archivo de cero; offset mayor que lo guardado -> 409 con los bytes que hay."""
    ext = ext.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, "Formato de video no permitido")
    if offset < 0 or total < 0:
        raise HTTPException(400, "Parametros invalidos")
    if total > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video demasiado grande")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    path = P(d, "video", "original" + ext)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if offset > have:
        return Response(json.dumps({"bytes": have}), status_code=409, media_type="application/json")
    size = offset
    with open(path, "wb" if offset == 0 else "r+b") as f:
        f.seek(offset)
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(413, "Video demasiado grande")   # el parcial se conserva
            f.write(chunk)
    with history_lock:
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
        meta["video"] = {"archivo": f"video/original{ext}", "bytes": os.path.getsize(path),
                         "total": total or None, "actualizado": now_iso()}
        jwrite(P(d, "metadata", "colonoscopy.json"), meta)
    sync_study(user, d, paciente, sesion)
    return {"ok": True, "bytes": meta["video"]["bytes"]}


DESCARTE_VENTANA = int(os.environ.get("DESCARTE_VENTANA_FRAMES", "50"))     # +-2 s a 25 fps


def upsert(items, frame, entry):
    items[:] = [x for x in items if x["frame"] != frame] + [entry]
    items.sort(key=lambda x: x["frame"])


@router.post("/api/findings")
async def save_finding(paciente: str = Form(...), sesion: str = Form(...), frame: int = Form(...),
                       tiempo: str = Form(""), tipo: str = Form(...), forma: str = Form(""),
                       box: str = Form(""), ancho: int = Form(0), alto: int = Form(0), umbral: float = Form(0),
                       thumb: UploadFile = File(...), frame_img: UploadFile = File(None),
                       mask: UploadFile = File(None), pred: UploadFile = File(None),
                       user: dict = Depends(current_user)):
    """automatico -> predictions/ (mascara de probabilidad + frame); manual -> annotations/ (mascara + caja)."""
    if tipo not in ("automatico", "manual") or frame < 0 or forma not in ("", "pincel", "caja", "mixta"):
        raise HTTPException(400, "Datos invalidos")
    try:
        bx = json.loads(box) if box else None
        if bx is not None and not (isinstance(bx, list) and len(bx) == 4 and all(isinstance(v, (int, float)) for v in bx)):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "Caja invalida")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    data = {"thumb": await thumb.read(), "frame": await frame_img.read() if frame_img else None,
            "mask": await mask.read() if mask else None, "pred": await pred.read() if pred else None}
    if any(b is not None and len(b) > MAX_BYTES for b in data.values()):
        raise HTTPException(413, "Archivo demasiado grande")
    n = f"{frame:06d}"
    auto = tipo == "automatico"
    paths = {"thumb": f"metadata/miniaturas/{tipo}_frame{n}.jpg",
             "frame": (f"predictions/frames/frame{n}.jpg" if auto else f"annotations/frames/frame{n}.jpg"),
             "mask": f"annotations/manual_masks/mask_frame{n}.png",
             "pred": f"predictions/masks/mask_frame{n}.png"}
    with history_lock:
        if auto and any(x.get("descartado") and abs(x["frame"] - frame) <= DESCARTE_VENTANA
                        for x in jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", [])):
            return {"ok": True, "ignorado": "descartado"}      # el medico ya descarto una deteccion aqui: no se vuelve a crear
        for k, rel in paths.items():
            if data[k]:
                with open(P(d, *rel.split("/")), "wb") as f:
                    f.write(data[k])
        box_r = [round(v) for v in bx] if bx else None
        if auto:
            fp = P(d, "predictions", "metadata.json")
            pm = jread(fp, {"predicciones": []})
            if ancho and alto:
                pm["ancho"], pm["alto"] = ancho, alto        # las cajas estan en pixeles de esta imagen
            upsert(pm.setdefault("predicciones", []), frame,
                   {"frame": frame, "tiempo": tiempo, "detectado": True, "box": box_r, "umbral": umbral or None,
                    "mascara": paths["pred"] if data["pred"] else None,
                    "frame_img": paths["frame"] if data["frame"] else None, "miniatura": paths["thumb"]})
            jwrite(fp, pm)
        else:
            fp = P(d, "annotations", "bounding_boxes.json")
            bb = jread(fp, {"cajas": []})
            if ancho and alto:
                bb["ancho"], bb["alto"] = ancho, alto
            upsert(bb.setdefault("cajas", []), frame,
                   {"frame": frame, "tiempo": tiempo, "forma": forma or "pincel", "box": box_r,
                    "mascara": paths["mask"] if data["mask"] else None,
                    "frame_img": paths["frame"] if data["frame"] else None, "miniatura": paths["thumb"]})
            jwrite(fp, bb)
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
        if ancho and alto:
            meta["ancho"], meta["alto"] = ancho, alto
            jwrite(P(d, "metadata", "colonoscopy.json"), meta)
        link_notes(d)
        refresh_notes_csv(d)
    sync_study(user, d, paciente, sesion)
    return {"ok": True}


@router.post("/api/findings/descartar")
async def discard_finding(paciente: str = Form(...), sesion: str = Form(...), frame: int = Form(...), tipo: str = Form(...),
                          restaurar: str = Form(""), user: dict = Depends(current_user)):
    """Descarta (o restaura) un hallazgo, p. ej. un falso positivo del modelo. No borra nada: el registro y sus imagenes quedan
    en disco marcados como 'descartado' (sirven como ejemplos negativos). Desaparece de la lista, del CSV y de los conteos.
    Es idempotente y no falla si el hallazgo no existe, para que la cola de envio del visor no se atasque."""
    if tipo not in ("automatico", "manual") or frame < 0:
        raise HTTPException(400, "Datos invalidos")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    fp, key = ((P(d, "predictions", "metadata.json"), "predicciones") if tipo == "automatico"
               else (P(d, "annotations", "bounding_boxes.json"), "cajas"))
    found = False
    with history_lock:
        data = jread(fp, {})
        for e in data.get(key, []):
            if e["frame"] == frame:
                found = True
                if restaurar:
                    e.pop("descartado", None)
                elif not e.get("descartado"):
                    e["descartado"] = {"fecha": now_iso(), "por": user.get("email", "")}
        if found:
            jwrite(fp, data)
            link_notes(d)
            refresh_notes_csv(d)
    if found:
        sync_study(user, d, paciente, sesion)
    return {"ok": True, "encontrado": found}


@router.post("/api/notes")
async def save_note(paciente: str = Form(...), sesion: str = Form(...), frame: int = Form(...),
                    tiempo: str = Form(""), texto: str = Form(...), origen: str = Form("texto"),
                    nota_id: str = Form(""), hallazgo: str = Form(""), user: dict = Depends(current_user)):
    """origen=voz -> transcriptions/transcript.json ; origen=texto -> annotations/descriptions.json.
    nota_id (generado por el visor) evita duplicados cuando un envio se reintenta."""
    texto = texto.strip()[:2000]
    if (not texto or frame < 0 or origen not in ("texto", "voz") or hallazgo not in ("", "automatico", "manual")
            or (nota_id and not re.match(r"^[A-Za-z0-9_-]{1,64}$", nota_id))):
        raise HTTPException(400, "Nota invalida")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    path, key = ((P(d, "transcriptions", "transcript.json"), "transcripciones") if origen == "voz"
                 else (P(d, "annotations", "descriptions.json"), "descripciones"))
    with history_lock:
        if nota_id and any(n.get("id") == nota_id for p, k in (
                (P(d, "transcriptions", "transcript.json"), "transcripciones"),
                (P(d, "annotations", "descriptions.json"), "descripciones")) for n in jread(p, {}).get(k, [])):
            return {"ok": True, "duplicada": True}
        data = jread(path, {})
        entry = {"frame": frame, "tiempo": tiempo, "texto": texto, "creado": now_iso()}
        if nota_id:
            entry["id"] = nota_id
        if hallazgo:                       # nota ligada a un hallazgo concreto (un frame puede tener uno automatico y uno manual)
            entry["hallazgo"] = hallazgo
        data.setdefault(key, []).append(entry)
        data[key].sort(key=lambda x: x["frame"])
        jwrite(path, data)
        link_notes(d)
        write_notes_csv(d)
    sync_study(user, d, paciente, sesion)
    return {"ok": True}


@router.get("/api/findings")
def list_findings(paciente: str, sesion: str, user: dict = Depends(current_user)):
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    url = lambda rel: f"api/findings/file?paciente={quote(paciente)}&sesion={quote(sesion)}&name={quote(rel)}"
    with history_lock:
        pred = jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", [])
        cajas = jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", [])
        notas = (jread(P(d, "annotations", "descriptions.json"), {}).get("descripciones", [])
                 + jread(P(d, "transcriptions", "transcript.json"), {}).get("transcripciones", []))
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
    mk_p = lambda p: {"frame": p["frame"], "tiempo": p["tiempo"], "tipo": "automatico", "forma": "",
                      "thumb": url(p["miniatura"]), "mask": None}
    mk_c = lambda c: {"frame": c["frame"], "tiempo": c["tiempo"], "tipo": "manual", "forma": c.get("forma", ""),
                      "thumb": url(c["miniatura"]), "mask": url(c["mascara"]) if c.get("mascara") else None}
    items = [mk_p(p) for p in pred if not p.get("descartado")] + [mk_c(c) for c in cajas if not c.get("descartado")]
    gone = ([{**mk_p(p), "descartado": p["descartado"]} for p in pred if p.get("descartado")]
            + [{**mk_c(c), "descartado": c["descartado"]} for c in cajas if c.get("descartado")])
    items.sort(key=lambda r: r["frame"])
    gone.sort(key=lambda r: r["frame"])
    return {"hallazgos": items, "descartados": gone, "ventana_descarte": DESCARTE_VENTANA,
            "notas": sorted(({"frame": n["frame"], "texto": n["texto"], "hallazgo": n.get("hallazgo", "")} for n in notas),
                            key=lambda n: n["frame"]),
            "tiene_video": video_completo(meta), "video": meta.get("video")}


@router.get("/api/sessions/estado")
def session_status(paciente: str, sesion: str, user: dict = Depends(current_user)):
    """Lo que el clúster tiene guardado de esta colonoscopia y el estado del envío de metadatos a Supabase."""
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    with history_lock:
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
        n_pred, n_caj, n_notas = study_counts(d)
    sb = meta.get("supabase") or {}
    estado = sb.get("estado", "pendiente") if SUPABASE_ACTIVO else "desactivado"
    if SUPABASE_ACTIVO and estado != "ok":
        sync_study(user, d, paciente, sesion)          # reintento en segundo plano
    vf = meta.get("video_final")
    if vf and vf.get("estado") == "procesando" and d not in _finalizing:
        vf = {**vf, "estado": "interrumpido"}       # el servidor se reinicio mientras preparaba el MP4: se puede volver a finalizar
    return {"video": meta.get("video"), "video_completo": video_completo(meta), "video_final": vf,
            "hallazgos": n_pred + n_caj, "notas": n_notas,
            "supabase": estado, "supabase_intento": sb.get("intento"), "supabase_detalle": sb.get("detalle")}


@router.get("/api/findings/file")
def finding_file(paciente: str, sesion: str, name: str, user: dict = Depends(current_user)):
    d = session_dir(user, paciente, sesion)
    if name == "hallazgos.csv":      # resumen generado al vuelo a partir de los JSON
        return Response(findings_csv_text(d), media_type="text/csv; charset=utf-8")
    if name == "notas.csv":
        if not os.path.isdir(d):
            raise HTTPException(404, "La colonoscopia no existe")
        with history_lock:
            body = notes_csv_text(d)
        return Response(body.encode("utf-8-sig"), media_type="text/csv; charset=utf-8")
    rel = os.path.normpath(name)
    path = os.path.realpath(os.path.join(d, rel))
    if rel.startswith("..") or os.path.isabs(rel) or not path.startswith(os.path.realpath(d) + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, "No existe")
    return FileResponse(path)


# --- conversion de video a un formato que el navegador reproduce ---
# El navegador solo reproduce bien MP4/H.264, 8 bits, yuv420p. Si el medico abre otro formato (AVI, MKV, HEVC, 10 bits...),
# el visor envia el archivo aqui y recibe una copia convertida SOLO para verla. El original se sigue guardando intacto
# en el historial por la subida normal. La conversion es con perdida (CRF 18): no uses la copia como archivo maestro.
CONVERT_TIMEOUT = int(os.environ.get("CONVERT_TIMEOUT_S", str(2 * 3600)))
_convert_slots = asyncio.Semaphore(int(os.environ.get("MAX_CONVERSIONS", "2")))


async def _run_ffmpeg(ffmpeg, src, dst, fps_flag, copy=False):
    if copy:                                                        # remux: sin recodificar, sin perdida y casi instantaneo
        vcodec = ["-c:v", "copy"]
    else:
        vcodec = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",       # yuv420p exige ancho y alto pares
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", *fps_flag]   # fps: conserva los tiempos originales
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-map", "0:v:0", *vcodec,
           "-an", "-movflags", "+faststart", dst]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), CONVERT_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(504, "La conversion tardo demasiado")
    return proc.returncode, err.decode("utf-8", "replace")


@router.post("/api/video/compatible")
async def video_compatible(request: Request, ext: str = ".bin", user: dict = Depends(current_user)):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(503, "FFmpeg no esta instalado en el servidor (o no esta en el PATH)")
    if not re.match(r"^\.[A-Za-z0-9]{2,5}$", ext):
        ext = ".bin"
    tmp = tempfile.mkdtemp(prefix="visor_conv_")
    keep = False
    try:
        src, dst = os.path.join(tmp, "entrada" + ext), os.path.join(tmp, "salida.mp4")
        size = 0
        with open(src, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(413, "Video demasiado grande")
                f.write(chunk)
        if not size:
            raise HTTPException(400, "Archivo vacio")
        async with _convert_slots:
            rc, err = await _run_ffmpeg(ffmpeg, src, dst, ["-fps_mode", "passthrough"])
            if rc != 0 and "fps_mode" in err:                       # FFmpeg anterior a 5.1
                rc, err = await _run_ffmpeg(ffmpeg, src, dst, ["-vsync", "0"])
        if rc != 0 or not os.path.isfile(dst) or not os.path.getsize(dst):
            raise HTTPException(422, "FFmpeg no pudo convertir el video: " + (err.strip()[-400:] or "sin detalle"))
        keep = True                                                 # se borra despues de enviar la respuesta
        return FileResponse(dst, media_type="video/mp4", background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
# --- fin conversion de video ---


# --- finalizar colonoscopia: video completo en MP4 + CSV ---
# Al terminar la camara el visor sube el ultimo trozo y llama aqui. El video que graba el navegador (webm, o mp4 fragmentado)
# no tiene duracion ni se puede navegar bien, asi que se deja una copia limpia en video/colonoscopia.mp4 (el original se conserva).
_finalizing = set()
_bg_tasks = set()


def _set_video_final(d, info):
    with history_lock:
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
        meta["video_final"] = info
        jwrite(P(d, "metadata", "colonoscopy.json"), meta)


async def _make_final_video(d, rel):
    src, dst = P(d, *rel.split("/")), P(d, "video", "colonoscopia.mp4")
    tmp = P(d, "video", "colonoscopia.part.mp4")
    inicio = now_iso()
    try:
        _set_video_final(d, {"estado": "procesando", "inicio": inicio})
        ffmpeg = shutil.which("ffmpeg")
        if not os.path.isfile(src) or not os.path.getsize(src):
            _set_video_final(d, {"estado": "sin_video", "inicio": inicio})
            return
        if not ffmpeg:
            _set_video_final(d, {"estado": "sin_ffmpeg", "inicio": inicio, "archivo": rel,
                                 "detalle": "FFmpeg no esta instalado en el servidor; el video quedo en su formato original"})
            return
        modos = (["copia"] if src.lower().endswith((".mp4", ".m4v", ".mov")) else []) + ["x264"]
        ok, err = False, ""
        for modo in modos:
            copy = modo == "copia"
            async with _convert_slots:
                rc, err = await _run_ffmpeg(ffmpeg, src, tmp, [] if copy else ["-fps_mode", "passthrough"], copy)
                if rc != 0 and "fps_mode" in err:                   # FFmpeg anterior a 5.1
                    rc, err = await _run_ffmpeg(ffmpeg, src, tmp, ["-vsync", "0"], copy)
            if rc == 0 and os.path.isfile(tmp) and os.path.getsize(tmp):
                ok = True
                break
        if ok:
            os.replace(tmp, dst)
            _set_video_final(d, {"estado": "ok", "inicio": inicio, "fin": now_iso(), "archivo": "video/colonoscopia.mp4",
                                 "bytes": os.path.getsize(dst), "modo": modo})
        else:
            _set_video_final(d, {"estado": "error", "inicio": inicio, "archivo": rel, "detalle": err.strip()[-300:] or "sin detalle"})
    except HTTPException as e:
        _set_video_final(d, {"estado": "error", "inicio": inicio, "archivo": rel, "detalle": str(e.detail)})
    except Exception as e:                                          # el original nunca se toca: siempre se puede reintentar
        _set_video_final(d, {"estado": "error", "inicio": inicio, "archivo": rel, "detalle": f"{type(e).__name__}: {e}"})
    finally:
        _finalizing.discard(d)
        if os.path.exists(tmp):
            os.remove(tmp)


@router.post("/api/sessions/finalizar")
async def finalize_session(paciente: str = Form(...), sesion: str = Form(...), user: dict = Depends(current_user)):
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    with history_lock:
        link_notes(d)
        write_notes_csv(d)                                          # notas.csv y hallazgos.csv quedan en la carpeta de la colonoscopia
        with open(P(d, "hallazgos.csv"), "w", encoding="utf-8-sig", newline="") as f:
            f.write(findings_csv_text(d))
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
    v = meta.get("video")
    if d in _finalizing:
        return {"ok": True, "estado": "procesando"}
    if not v:
        _set_video_final(d, {"estado": "sin_video"})
        return {"ok": True, "estado": "sin_video"}
    _finalizing.add(d)                                              # se registra antes de responder: el visor consulta el estado enseguida
    _set_video_final(d, {"estado": "procesando", "inicio": now_iso()})
    task = asyncio.create_task(_make_final_video(d, v["archivo"]))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return {"ok": True, "estado": "procesando"}
# --- fin finalizar colonoscopia ---


# --- videos guardados: lista, estado de revision y reproduccion ---
# Las colonoscopias se graban/suben primero y el medico las revisa despues. El <video> del navegador no puede mandar el token de
# login, asi que el visor pide un "ticket" (aleatorio, vence en 8 h, solo sirve para ese archivo) y reproduce api/video/<ticket>.
# El servidor responde por rangos (206): se puede avanzar y retroceder sin descargar todo el video.
REVISION_ESTADOS = ("pendiente", "en_revision", "revisada")
PLAYABLE = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm"}      # lo que el navegador reproduce de forma fiable
TICKET_TTL = 8 * 3600
_tickets = {}


def _study_dirs(user):
    root = user_dir(user)
    if not os.path.isdir(root):
        return
    for pn in sorted(os.listdir(root)):
        pdir = os.path.join(root, pn)
        paciente = pn[len("paciente_"):]
        if not (pn.startswith("paciente_") and PID_RE.match(paciente) and os.path.isdir(pdir)):
            continue
        for sn in sorted(os.listdir(pdir)):
            d = os.path.join(pdir, sn)
            if SESION_RE.match(sn) and os.path.isdir(d):
                yield paciente, sn, d


@router.get("/api/studies")
def list_studies(user: dict = Depends(current_user)):
    """Todas las colonoscopias de la cuenta (de todos los pacientes), la mas reciente primero."""
    out = []
    with history_lock:
        for paciente, sesion, d in _study_dirs(user):
            meta = jread(P(d, "metadata", "colonoscopy.json"), {})
            n_pred, n_caj, n_notas = study_counts(d)
            v, rev = meta.get("video") or {}, meta.get("revision") or {}
            out.append({"paciente": paciente, "sesion": sesion, "creado": meta.get("creado", ""), "fuente": meta.get("fuente", ""),
                        "hallazgos": n_pred + n_caj, "notas": n_notas,
                        "video": "listo" if video_completo(meta) else ("incompleto" if v else "ninguno"),
                        "video_bytes": v.get("bytes"),
                        "revision": rev.get("estado", "pendiente"), "revision_fecha": rev.get("fecha"), "revision_por": rev.get("por")})
    out.sort(key=lambda r: r["creado"] or "", reverse=True)
    return out


@router.post("/api/studies/revision")
def set_revision(paciente: str = Form(...), sesion: str = Form(...), estado: str = Form(...), user: dict = Depends(current_user)):
    if estado not in REVISION_ESTADOS:
        raise HTTPException(400, "Estado de revision invalido")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    with history_lock:
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
        meta["revision"] = {"estado": estado, "fecha": now_iso(), "por": user.get("email", "")}
        jwrite(P(d, "metadata", "colonoscopy.json"), meta)
    return meta["revision"]


@router.post("/api/sessions/video-ticket")
def video_ticket(paciente: str = Form(...), sesion: str = Form(...), user: dict = Depends(current_user)):
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    with history_lock:
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
    v = meta.get("video")
    final = P(d, "video", "colonoscopia.mp4")
    final = final if (meta.get("video_final") or {}).get("estado") == "ok" and os.path.isfile(final) else None
    original = P(d, *v["archivo"].split("/")) if v and video_completo(meta) else None
    path = final or (original if original and os.path.isfile(original) else None)
    if not path:
        raise HTTPException(404, "Esta colonoscopia no tiene un video guardado completo")
    ext = os.path.splitext(path)[1].lower()
    now = time.time()
    for k in [k for k, t in _tickets.items() if t["exp"] < now]:
        del _tickets[k]
    while len(_tickets) >= 500:
        _tickets.pop(next(iter(_tickets)))
    tk = secrets.token_urlsafe(24)
    _tickets[tk] = {"path": path, "exp": now + TICKET_TTL, "mime": PLAYABLE.get(ext, "application/octet-stream")}
    return {"url": f"api/video/{tk}", "formato": ext, "reproducible": ext in PLAYABLE, "convertido": path == final,
            "bytes": os.path.getsize(path)}


def _byte_range(header, size):
    """'bytes=a-b' | 'bytes=a-' | 'bytes=-n' -> (inicio, fin) inclusivos; None si no es un rango; ValueError si no se puede satisfacer."""
    m = re.match(r"^bytes=(\d*)-(\d*)$", (header or "").strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None
    if not m.group(1):
        start, end = max(0, size - int(m.group(2))), size - 1
    else:
        start, end = int(m.group(1)), min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
    if start > end or start >= size:
        raise ValueError
    return start, end


def _iter_file(path, start, end, chunk=1 << 20):
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            b = f.read(min(chunk, left))
            if not b:
                break
            left -= len(b)
            yield b


@router.get("/api/video/{ticket}")
def stream_video(ticket: str, request: Request):
    t = _tickets.get(ticket)
    if not t or t["exp"] < time.time() or not os.path.isfile(t["path"]):
        raise HTTPException(404, "El enlace del video vencio; vuelve a abrir la colonoscopia")
    size = os.path.getsize(t["path"])
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, no-store"}
    try:
        rng = _byte_range(request.headers.get("range"), size)
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if rng:
        start, end = rng
        headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})
        return StreamingResponse(_iter_file(t["path"], start, end), status_code=206, media_type=t["mime"], headers=headers)
    headers["Content-Length"] = str(size)
    return StreamingResponse(_iter_file(t["path"], 0, size - 1), media_type=t["mime"], headers=headers)
# --- fin videos guardados ---


@router.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


# ---------- montaje: raiz y prefijos ----------
# BASE_PATHS = prefijos separados por coma bajo los que tambien se sirve el visor.
# Por defecto: /projects/colon_viewe  (y /projects/colon_viewer, por si el nombre lleva la "r").
BASE_PATHS = ["/" + p.strip().strip("/")
              for p in os.environ.get("BASE_PATHS", "/projects/colon_viewe,/projects/colon_viewer").split(",")
              if p.strip().strip("/")]
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _redirect_to(path):
    return lambda: RedirectResponse(path + "/")   # sin la barra final, las URL relativas del visor se romperian


app.include_router(router)                         # http://localhost:8000/
for _bp in BASE_PATHS:
    app.include_router(router, prefix=_bp)         # http://localhost:8000/projects/colon_viewe/
    app.add_api_route(_bp, _redirect_to(_bp), methods=["GET"], include_in_schema=False)
    app.mount(_bp + "/static", StaticFiles(directory=STATIC_DIR), name="static" + _bp.replace("/", "_"))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
