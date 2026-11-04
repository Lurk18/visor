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
import os
import re
import json
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import quote

import cv2
import httpx
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask
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
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2 GiB por video
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
HISTORY_DIR = os.environ.get("HISTORY_DIR", "./historial")
HEAD = ["numero_hallazgo", "frame", "tiempo", "tipo", "archivo_imagen", "archivo_mascara"]
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


@app.get("/api/config")
def public_config():
    """Datos publicos que necesita el navegador para iniciar sesion."""
    return {"supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(state.get("device")), "model_loaded": "model" in state,
            "supabase": state.get("supabase")}


@app.post("/api/video/compatible")
async def compatible_video(file: UploadFile = File(...), user: dict = Depends(current_user)):
    """Convierte un video subido a MP4/H.264/yuv420p para reproduccion web.

    El original no se guarda: se procesa en un directorio temporal y se elimina
    cuando FastAPI termina de enviar el MP4 convertido al navegador.
    """
    if not shutil.which(FFMPEG_BIN):
        raise HTTPException(503, "FFmpeg no esta instalado en el servidor. Instala FFmpeg y reinicia el visor.")

    temp_dir = tempfile.mkdtemp(prefix="visor_video_")
    input_path = os.path.join(temp_dir, "original_upload")
    output_path = os.path.join(temp_dir, "video_compatible.mp4")
    total = 0
    try:
        # Escritura por bloques para no cargar videos grandes completos en RAM.
        with open(input_path, "wb") as dst:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise HTTPException(413, f"El video supera el limite de {MAX_VIDEO_BYTES // (1024 * 1024)} MB.")
                dst.write(chunk)
        if total == 0:
            raise HTTPException(400, "El archivo de video esta vacio.")

        command = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-map", "0:v:0", "-an",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-f", "mp4", output_path,
        ]
        result = await asyncio.to_thread(
            subprocess.run, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=7200, check=False
        )
        if result.returncode != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            detail = (result.stderr or "FFmpeg no pudo convertir el archivo.")[-1800:]
            raise HTTPException(422, f"No se pudo convertir el video: {detail}")

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "video")
        download_name = os.path.splitext(safe_name)[0] + "_compatible.mp4"
        return FileResponse(
            output_path, media_type="video/mp4", filename=download_name,
            background=BackgroundTask(shutil.rmtree, temp_dir, ignore_errors=True),
        )
    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(504, "La conversion excedio el tiempo limite de 2 horas.")
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(500, f"Error durante la conversion: {type(exc).__name__}")
    finally:
        await file.close()


@app.post("/api/segment")
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


@app.websocket("/ws")
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


# ---------- historial de hallazgos (mismo formato CSV que el visor de escritorio) ----------
def video_dir(uid: str, video: str) -> str:
    uid = re.sub(r"[^A-Za-z0-9_-]", "_", uid)[:64] or "anon"
    key = re.sub(r"[^A-Za-z0-9._-]", "_", video)[:80].strip(".") or "video"
    return os.path.join(HISTORY_DIR, uid, key + "_hallazgos")


def read_rows(d):
    p = os.path.join(d, "hallazgos.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(d, rows):
    with open(os.path.join(d, "hallazgos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["frame"], r["tiempo"], r["tipo"], r["archivo_imagen"], r["archivo_mascara"]])


@app.post("/api/findings")
async def save_finding(video: str = Form(...), frame: int = Form(...), tiempo: str = Form(""),
                       tipo: str = Form(...), thumb: UploadFile = File(...),
                       mask: UploadFile = File(None), user: dict = Depends(current_user)):
    if tipo not in ("automatico", "manual") or frame < 0:
        raise HTTPException(400, "Datos invalidos")
    thumb_b = await thumb.read()
    mask_b = await mask.read() if mask else None
    if len(thumb_b) > MAX_BYTES or (mask_b and len(mask_b) > MAX_BYTES):
        raise HTTPException(413, "Archivo demasiado grande")
    d = video_dir(user["id"], video)
    img_name = f"hallazgo_{tipo}_frame{frame:06d}.jpg"
    mask_name = f"mascara_frame{frame:06d}.png" if mask_b else ""
    with history_lock:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, img_name), "wb") as f:
            f.write(thumb_b)
        if mask_b:
            with open(os.path.join(d, mask_name), "wb") as f:
                f.write(mask_b)
        rows = [r for r in read_rows(d) if not (int(r["frame"]) == frame and r["tipo"] == tipo)]
        rows.append({"frame": frame, "tiempo": tiempo, "tipo": tipo,
                     "archivo_imagen": img_name, "archivo_mascara": mask_name})
        rows.sort(key=lambda r: int(r["frame"]))
        write_rows(d, rows)
    return {"ok": True, "total": len(rows)}


@app.get("/api/findings")
def list_findings(video: str, user: dict = Depends(current_user)):
    d = video_dir(user["id"], video)
    url = lambda n: f"/api/findings/file?video={quote(video)}&name={n}"
    with history_lock:
        rows = read_rows(d)
    return [{"frame": int(r["frame"]), "tiempo": r["tiempo"], "tipo": r["tipo"],
             "thumb": url(r["archivo_imagen"]),
             "mask": url(r["archivo_mascara"]) if r["archivo_mascara"] else None} for r in rows]


@app.get("/api/findings/file")
def finding_file(video: str, name: str, user: dict = Depends(current_user)):
    path = os.path.join(video_dir(user["id"], video), os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "No existe")
    return FileResponse(path)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
