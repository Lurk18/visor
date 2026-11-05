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
import io
import os
import re
import json
import threading
import time
from urllib.parse import quote

import cv2
import httpx
import numpy as np
import torch
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

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
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
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
    user = await verify_token(authorization.removeprefix("Bearer ").strip())
    if not user:
        raise HTTPException(401, "No autenticado")
    return user


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
        frames.setdefault(p["frame"], []).append("automatico")
    for c in jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", []):
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
                            "hallazgos": len(jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", []))
                                         + len(jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", [])),
                            "tiene_video": bool(meta.get("video"))})
    return out


@router.put("/api/sessions/video")
async def upload_video(request: Request, paciente: str, sesion: str, ext: str = ".mp4",
                       append: int = 0, user: dict = Depends(current_user)):
    """Guarda video/original.<ext> (cuerpo crudo; append=1 para la grabacion de la camara por trozos)."""
    ext = ext.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, "Formato de video no permitido")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    path = P(d, "video", "original" + ext)
    size = os.path.getsize(path) if (append and os.path.exists(path)) else 0
    try:
        with open(path, "ab" if append else "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(413, "Video demasiado grande")
                f.write(chunk)
    except HTTPException:
        if not append and os.path.exists(path):
            os.remove(path)
        raise
    with history_lock:
        meta = jread(P(d, "metadata", "colonoscopy.json"), {})
        meta["video"] = {"archivo": f"video/original{ext}", "bytes": size, "actualizado": now_iso()}
        jwrite(P(d, "metadata", "colonoscopy.json"), meta)
    return {"ok": True, "bytes": size}


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
    return {"ok": True}


@router.post("/api/notes")
async def save_note(paciente: str = Form(...), sesion: str = Form(...), frame: int = Form(...),
                    tiempo: str = Form(""), texto: str = Form(...), origen: str = Form("texto"),
                    user: dict = Depends(current_user)):
    """origen=voz -> transcriptions/transcript.json ; origen=texto -> annotations/descriptions.json"""
    texto = texto.strip()[:2000]
    if not texto or frame < 0 or origen not in ("texto", "voz"):
        raise HTTPException(400, "Nota invalida")
    d = session_dir(user, paciente, sesion)
    if not os.path.isdir(d):
        raise HTTPException(404, "La colonoscopia no existe")
    path, key = ((P(d, "transcriptions", "transcript.json"), "transcripciones") if origen == "voz"
                 else (P(d, "annotations", "descriptions.json"), "descripciones"))
    with history_lock:
        data = jread(path, {})
        data.setdefault(key, []).append({"frame": frame, "tiempo": tiempo, "texto": texto, "creado": now_iso()})
        data[key].sort(key=lambda x: x["frame"])
        jwrite(path, data)
        link_notes(d)
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
    items = ([{"frame": p["frame"], "tiempo": p["tiempo"], "tipo": "automatico", "forma": "",
               "thumb": url(p["miniatura"]), "mask": None} for p in pred]
             + [{"frame": c["frame"], "tiempo": c["tiempo"], "tipo": "manual", "forma": c.get("forma", ""),
                 "thumb": url(c["miniatura"]), "mask": url(c["mascara"]) if c.get("mascara") else None} for c in cajas])
    items.sort(key=lambda r: r["frame"])
    return {"hallazgos": items,
            "notas": sorted(({"frame": n["frame"], "texto": n["texto"]} for n in notas), key=lambda n: n["frame"]),
            "tiene_video": bool(meta.get("video"))}


@router.get("/api/findings/file")
def finding_file(paciente: str, sesion: str, name: str, user: dict = Depends(current_user)):
    d = session_dir(user, paciente, sesion)
    if name == "hallazgos.csv":      # resumen generado al vuelo a partir de los JSON
        rows = [["frame", "tiempo", "tipo", "forma", "x", "y", "ancho_caja", "alto_caja", "mascara", "frame_img"]]
        for p in jread(P(d, "predictions", "metadata.json"), {}).get("predicciones", []):
            b = p.get("box") or ["", "", "", ""]
            rows.append([p["frame"], p["tiempo"], "automatico", "", *b, p.get("mascara") or "", p.get("frame_img") or ""])
        for c in jread(P(d, "annotations", "bounding_boxes.json"), {}).get("cajas", []):
            b = c.get("box") or ["", "", "", ""]
            rows.append([c["frame"], c["tiempo"], "manual", c.get("forma", ""), *b, c.get("mascara") or "", c.get("frame_img") or ""])
        body = io.StringIO()
        csv.writer(body).writerows(rows)
        return Response(body.getvalue(), media_type="text/csv; charset=utf-8")
    rel = os.path.normpath(name)
    path = os.path.realpath(os.path.join(d, rel))
    if rel.startswith("..") or os.path.isabs(rel) or not path.startswith(os.path.realpath(d) + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, "No existe")
    return FileResponse(path)


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
