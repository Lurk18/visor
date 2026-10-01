"""
Visor en tiempo real — UACANet-v2 (colonoscopia)  [multi-video + anotacion manual]
====================================================================================

Muestra en paralelo:
    [ video original ]   [ overlay de segmentacion ]   [ panel: estado + historial ]

Novedades de esta version:
    - Se puede cargar una CARPETA con varios videos (--library) y elegir cual ver;
      dentro del visor, con 'n'/'p' (o los botones <VID / VID>) se cambia de video
      sin cerrar el programa.
    - Si un video ya tiene un historial guardado de una sesion anterior (carpeta
      "<video>_hallazgos"), se carga automaticamente al abrirlo: tanto los hallazgos
      automaticos como las marcas manuales reaparecen.
    - Modo de anotacion manual ('e' o boton ANOTAR): con el mouse, clic izquierdo y
      arrastrar dibuja una marca (pincel) sobre el panel de segmentacion; clic derecho
      borra. 'g' (o boton GUARDAR) guarda esa marca en el historial, igual que un
      hallazgo automatico pero etiquetado como manual. Util para que el doctor marque
      el polipo el mismo, sin depender del modelo.

Controles mientras corre en tu PC:
    ESPACIO / click PLAY-PAUSA : pausar o reanudar
    STOP / 's'                 : volver al frame 0
    <1  o flecha izq / 'a'     : retroceder 1 frame
    <<10 o 'z'                 : retroceder 10 frames
    1>  o flecha der / 'd'     : avanzar 1 frame
    10>> o 'x'                 : avanzar 10 frames
    -THR / +THR  o '-' / '+'   : bajar/subir el umbral de binarizacion
    slider "Umbral x100"       : mismo umbral, de forma continua
    slider "Posicion"          : arrastrar para saltar a cualquier frame
    ANALISIS / 'v'             : prender/apagar la inferencia del modelo
    MODO / 'm'                 : alternar entre mascara rellena y solo bbox
    ANOTAR / 'e'                : prender/apagar el modo de marca manual (pausa el video)
       clic izq + arrastrar en el panel de segmentacion : marcar (pincel amarillo)
       clic der + arrastrar                              : borrar
       '[' / ']'                                          : pincel mas chico / grande
       'c'                                                 : borrar la marca manual del frame actual
    GUARDAR / 'g'               : guardar la marca manual actual en el historial
    <VID / 'p'                  : video anterior de la lista (si hay varios)
    VID> / 'n'                  : video siguiente de la lista (si hay varios)
    rueda del mouse sobre el panel : desplaza el historial de hallazgos
    clic en una miniatura           : salta a ese frame
    SALIR / 'q' / ESC           : cerrar (guarda el historial del video actual)

El retroceso usa un buffer en memoria de los ultimos N frames (--cache-frames) en vez
de re-decodificar el .avi hacia atras, que es justo donde los archivos MJPEG se rompen.
El slider de posicion sigue la misma logica: si el frame pedido ya esta en el buffer
salta directo; si esta mas adelante, avanza sin decodificar (cap.grab, barato); si esta
mas atras del buffer, reabre el video y vuelve a leer secuencialmente desde el inicio.

Uso tipico:

    # Un solo video, con controles:
    python viewer_uacanet.py --source video.avi --checkpoint best_model.pth

    # Una carpeta con varios videos: elige cual ver, navega con n/p adentro:
    python viewer_uacanet.py --library C:\\videos_paciente --checkpoint best_model.pth

    # Camara en vivo (sin navegacion entre videos, un solo "video" = la camara):
    python viewer_uacanet.py --source 0 --checkpoint best_model.pth

    # Kaggle / sin GUI, guardando a archivo:
    python viewer_uacanet.py --source video.mp4 --checkpoint best_model.pth \
        --output /kaggle/working/demo.mp4 --no-display

Requisitos: torch, torchvision, timm, opencv-python, numpy
"""

import argparse
import csv
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False
    print("Advertencia: timm no esta instalado. Ejecuta: pip install timm")


# =========================================================
# 1. PREPROCESADO (igual que en entrenamiento)
# =========================================================
def auto_crop_endoscopy(img_np, threshold=15, min_area_frac=0.05):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    k = max(3, int(min(img_np.shape[:2]) * 0.03))
    kernel = np.ones((k, k), np.uint8)
    cleaned = cv2.erode(binary, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, img_np.shape[0], 0, img_np.shape[1]

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area_frac * img_np.shape[0] * img_np.shape[1]:
        return 0, img_np.shape[0], 0, img_np.shape[1]

    x, y, w, h = cv2.boundingRect(largest)
    return y, y + h, x, x + w


def resize_with_padding(img, desired_size=352, is_mask=False):
    channels = 1 if img.ndim == 2 else img.shape[2]
    h, w = img.shape[:2]
    ratio = desired_size / max(h, w)
    new_h, new_w = max(1, int(h * ratio)), max(1, int(w * ratio))

    inter = cv2.INTER_NEAREST if channels == 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=inter)

    top = (desired_size - new_h) // 2
    bottom = desired_size - new_h - top
    left = (desired_size - new_w) // 2
    right = desired_size - new_w - left

    border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT
    value = 0 if is_mask else None
    if is_mask:
        canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, border, value=value)
    else:
        canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, border)

    return canvas, (top, bottom, left, right, new_h, new_w)


# =========================================================
# 2. UACA ADAPTIVE + UACANET-V2 (identico al de entrenamiento)
# =========================================================
class UACAAdaptive(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.psi = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.phi = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.omega = nn.Conv2d(in_channels, in_channels, 1)
        self.delta = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.logit_tau_f = nn.Parameter(torch.zeros(1))
        self.logit_tau_b = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _norm(mm):
        s = mm.view(mm.size(0), -1).sum(1).view(-1, 1, 1, 1)
        s = torch.clamp(s, min=1e-6)
        return mm / s

    def forward(self, x, m):
        tau_f = torch.sigmoid(self.logit_tau_f)
        tau_b = torch.sigmoid(self.logit_tau_b)

        m_f = F.relu(m - tau_f)
        m_b = F.relu(tau_b - m)
        m_u = 0.5 - torch.abs(m - 0.5)
        m_f, m_b, m_u = self._norm(m_f), self._norm(m_b), self._norm(m_u)

        v_f = torch.sum(x * m_f, dim=(2, 3), keepdim=True)
        v_b = torch.sum(x * m_b, dim=(2, 3), keepdim=True)
        v_u = torch.sum(x * m_u, dim=(2, 3), keepdim=True)

        psi_x = self.psi(x)
        phi_f, phi_b, phi_u = self.phi(v_f), self.phi(v_b), self.phi(v_u)

        s_f = torch.clamp(torch.sum(psi_x * phi_f, dim=1, keepdim=True), -30, 30)
        s_b = torch.clamp(torch.sum(psi_x * phi_b, dim=1, keepdim=True), -30, 30)
        s_u = torch.clamp(torch.sum(psi_x * phi_u, dim=1, keepdim=True), -30, 30)

        S = torch.exp(s_f) + torch.exp(s_b) + torch.exp(s_u) + 1e-6
        sf, sb, su = torch.exp(s_f) / S, torch.exp(s_b) / S, torch.exp(s_u) / S

        vf, vb, vu = self.omega(v_f), self.omega(v_b), self.omega(v_u)
        t = sf * vf + sb * vb + su * vu

        out = torch.cat([x, t], dim=1)
        out = self.delta(out)
        return torch.nan_to_num(out)


class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UACANetV2(nn.Module):
    def __init__(self, out_channels=1, pretrained=False):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("Instala timm: pip install timm")

        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        feats = [96, 192, 384, 768]

        self.dec3 = SimpleDecoderBlock(feats[3], 256)
        self.dec2 = SimpleDecoderBlock(256 + feats[2], 128)
        self.dec1 = SimpleDecoderBlock(128 + feats[1], 64)
        self.dec0 = SimpleDecoderBlock(64 + feats[0], 64)

        self.uaca_mid = UACAAdaptive(128)
        self.uaca_small = UACAAdaptive(64)
        self.reduce_mid = nn.Conv2d(128, 64, 1)

        self.sal_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, out_channels, 1),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x):
        f0, f1, f2, f3 = self.backbone(x)

        d3 = self.dec3(f3)
        d3 = F.interpolate(d3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3, f2], dim=1))
        d2u = F.interpolate(d2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2u, f1], dim=1))
        d1u = F.interpolate(d1, size=f0.shape[2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([d1u, f0], dim=1))

        sal_init = torch.sigmoid(self.sal_head(d0))

        uaca_mid_in = F.interpolate(d2, size=sal_init.shape[2:], mode="bilinear", align_corners=False)
        uaca_mid_out = self.uaca_mid(uaca_mid_in, sal_init)
        uaca_small_out = self.uaca_small(d0, sal_init)

        uaca_mid_out = F.interpolate(
            uaca_mid_out, size=uaca_small_out.shape[2:], mode="bilinear", align_corners=False
        )
        uaca_mid_out = self.reduce_mid(uaca_mid_out)
        fused = uaca_small_out + uaca_mid_out

        out = torch.sigmoid(self.final_conv(fused))
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# =========================================================
# 3. CARGA DE MODELO
# =========================================================
def load_model(checkpoint_path, device):
    model = UACANetV2(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modelo cargado desde {checkpoint_path}")
    if "val_iou" in ckpt:
        print(f"  IoU de validacion registrado (entrenamiento): {ckpt['val_iou']:.4f}")
    return model


# =========================================================
# 3b. TIMECODE -> INDICE DE FRAME
# =========================================================
def parse_timecode(ts, fps):
    parts = [float(p) for p in ts.strip().split(":")]

    if len(parts) == 1:
        total_seconds = parts[0]
    elif len(parts) == 2:
        mm, ss = parts
        total_seconds = mm * 60 + ss
    elif len(parts) == 3:
        a, b, c = parts
        if c < fps:
            mm, ss, ff = a, b, c
            total_seconds = mm * 60 + ss + ff / fps
        else:
            hh, mm, ss = a, b, c
            total_seconds = hh * 3600 + mm * 60 + ss
    else:
        raise ValueError(f"Formato de timecode no reconocido: {ts}")

    return int(round(total_seconds * fps))


def seconds_to_mmss(seconds):
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


# =========================================================
# 4. INFERENCIA POR FRAME
# =========================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def segment_frame(model, frame_bgr, device, size=352, use_autocast=True):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    y0, y1, x0, x1 = auto_crop_endoscopy(frame_rgb)
    crop = frame_rgb[y0:y1, x0:x1]

    padded, (top, bottom, left, right, new_h, new_w) = resize_with_padding(crop, size, is_mask=False)

    inp = padded.astype(np.float32) / 255.0
    inp = (inp - IMAGENET_MEAN) / IMAGENET_STD
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    if use_autocast and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(inp)
    else:
        pred = model(inp)

    pred = pred.squeeze().float().cpu().numpy()

    pred_crop = pred[top: top + new_h, left: left + new_w]
    pred_crop = cv2.resize(pred_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

    full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
    full_mask[y0:y1, x0:x1] = pred_crop

    return full_mask, (y0, y1, x0, x1)


def make_overlay(frame_bgr, mask_prob, threshold=0.5, color=(0, 0, 255), alpha=0.45,
                 mode="mask", min_comp_frac=2e-4):
    """
    mode="mask": relleno rojo translucido + contorno verde.
    mode="bbox": sin relleno; solo el rectangulo de cada region detectada.
    """
    binary_mask = (mask_prob > threshold).astype(np.uint8)

    if mode == "bbox":
        overlay = frame_bgr.copy()
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
        min_area = max(20, int(min_comp_frac * binary_mask.size))
        for i in range(1, n_lab):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
        return overlay, binary_mask

    color_layer = np.zeros_like(frame_bgr)
    color_layer[:] = color
    mask3 = binary_mask[:, :, None]
    overlay = np.where(mask3 == 1, cv2.addWeighted(frame_bgr, 1 - alpha, color_layer, alpha, 0), frame_bgr)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay, binary_mask


# =========================================================
# 5. DETECCION DE REGIONES (solo para saber SI hay polipo)
# =========================================================
def analyze_regions(binary, min_comp_frac=2e-4):
    total = binary.size
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(20, int(min_comp_frac * total))
    comps = [i for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    bbox = None
    if comps:
        big = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        bbox = (int(stats[big, cv2.CC_STAT_LEFT]), int(stats[big, cv2.CC_STAT_TOP]),
                int(stats[big, cv2.CC_STAT_WIDTH]), int(stats[big, cv2.CC_STAT_HEIGHT]))

    return {"n_regions": len(comps), "bbox": bbox, "has_detection": len(comps) > 0}


def make_finding_thumbnail(overlay_bgr, bbox, thumb_w=150, thumb_h=100, margin_frac=0.6):
    h, w = overlay_bgr.shape[:2]
    if bbox is None:
        crop = overlay_bgr
    else:
        x, y, bw, bh = bbox
        mx = int(bw * margin_frac) + 15
        my = int(bh * margin_frac) + 15
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        crop = overlay_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            crop = overlay_bgr
    return cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)


# =========================================================
# 6. GUARDAR / CARGAR HISTORIAL (automatico + manual)
# =========================================================
def default_history_dir(source):
    if isinstance(source, str) and os.path.exists(source):
        base = os.path.splitext(os.path.basename(source))[0]
        folder = os.path.dirname(os.path.abspath(source)) or "."
        return os.path.join(folder, f"{base}_hallazgos")
    return os.path.join(".", "hallazgos_camara")


def save_history(findings, out_dir):
    if not findings:
        return
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "hallazgos.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["numero_hallazgo", "frame", "tiempo", "tipo", "archivo_imagen", "archivo_mascara"])
        for i, item in enumerate(findings, start=1):
            img_name = f"hallazgo_{i:03d}_frame{item['idx']:06d}.png"
            cv2.imwrite(os.path.join(out_dir, img_name), item["thumb"])
            mask_name = ""
            if item.get("mask") is not None:
                mask_name = f"mascara_frame{item['idx']:06d}.png"
                cv2.imwrite(os.path.join(out_dir, mask_name), item["mask"])
            writer.writerow([i, item["idx"], seconds_to_mmss(item["time_s"]),
                             item.get("tipo", "automatico"), img_name, mask_name])
    print(f"\nHistorial guardado: {len(findings)} hallazgo(s) en '{out_dir}'.")


def load_history(history_dir, fps):
    """Lee un historial guardado en una sesion anterior (si existe) y lo
    reconstruye: findings (para el panel) + manual_annotations (para poder
    seguir editando las marcas manuales sobre el video)."""
    findings = []
    manual_annotations = {}
    csv_path = os.path.join(history_dir, "hallazgos.csv")
    if not os.path.exists(csv_path):
        return findings, manual_annotations
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["frame"])
                img_name = row.get("archivo_imagen", "")
                thumb_path = os.path.join(history_dir, img_name) if img_name else None
                if thumb_path and os.path.exists(thumb_path):
                    thumb = cv2.imread(thumb_path)
                else:
                    thumb = np.zeros((100, 150, 3), dtype=np.uint8)
                tipo = row.get("tipo") or "automatico"
                item = {"idx": idx, "time_s": idx / fps if fps else 0.0, "thumb": thumb, "tipo": tipo}
                mask_name = row.get("archivo_mascara", "")
                if mask_name:
                    mpath = os.path.join(history_dir, mask_name)
                    if os.path.exists(mpath):
                        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            item["mask"] = m
                            manual_annotations[idx] = m
                findings.append(item)
    except Exception as e:
        print(f"Aviso: no se pudo leer el historial previo ({e}); se empieza vacio.")
        return [], {}
    return findings, manual_annotations


# =========================================================
# 7. UTILIDAD DE DIBUJO: pegar una imagen recortando lo que no cabe
# =========================================================
def paste(dst, src, x, y):
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = x, y, x + w, y + h
    sx0, sy0, sx1, sy1 = 0, 0, w, h
    if x0 < 0:
        sx0 = -x0
        x0 = 0
    if y0 < 0:
        sy0 = -y0
        y0 = 0
    if x1 > W:
        sx1 -= (x1 - W)
        x1 = W
    if y1 > H:
        sy1 -= (y1 - H)
        y1 = H
    if x0 >= x1 or y0 >= y1 or sx0 >= sx1 or sy0 >= sy1:
        return
    dst[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]


# =========================================================
# 8. PANEL LATERAL: ESTADO (arriba) + HISTORIAL (abajo, con scroll)
# =========================================================
PANEL_W = 300
HEADER_H = 205          # alto de la seccion de estado; el resto es el historial
THUMB_W, THUMB_H = 150, 100
ITEM_H = THUMB_H + 34
BAR_H = 64
SEP_W = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

BUTTONS_SPEC = [
    ("STOP", "stop"), ("<<10", "back10"), ("<1", "back1"), ("PLAY/PAUSA", "toggle"),
    ("1>", "fwd1"), ("10>>", "fwd10"), ("-THR", "thr_down"), ("+THR", "thr_up"),
    ("ANALISIS", "toggle_analysis"), ("MODO", "toggle_mode"),
    ("ANOTAR", "toggle_manual"), ("GUARDAR", "save_manual"),
    ("<VID", "prev_video"), ("VID>", "next_video"), ("SALIR", "quit"),
]


def draw_status_header(width, height, frame_idx, t_s, paused, threshold, render_mode,
                       analysis_on, n_findings, brush_mode=False, brush_radius=15, video_label=""):
    header = np.full((height, width, 3), 25, dtype=np.uint8)
    margin = 14
    fs = 0.48
    y = 20
    if video_label:
        cv2.putText(header, video_label, (margin, y), FONT, 0.42, (150, 200, 255), 1, cv2.LINE_AA)
        y += 20
    cv2.putText(header, "ESTADO", (margin, y), FONT, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    y += 24
    cv2.putText(header, f"Frame {frame_idx}   t={seconds_to_mmss(t_s)}", (margin, y),
                FONT, fs, (225, 225, 225), 1, cv2.LINE_AA)
    y += 20
    state_txt = "PAUSA" if paused else "REPRODUCIENDO"
    state_col = (0, 200, 255) if paused else (120, 255, 120)
    cv2.putText(header, f"Estado: {state_txt}", (margin, y), FONT, fs, state_col, 1, cv2.LINE_AA)
    y += 20
    modo_txt = "Mascara" if render_mode == "mask" else "BBox"
    cv2.putText(header, f"Umbral: {threshold:.2f}   Modo: {modo_txt}", (margin, y),
                FONT, fs, (225, 225, 225), 1, cv2.LINE_AA)
    y += 20
    if not analysis_on:
        cv2.putText(header, "Analisis: DESACTIVADO", (margin, y), FONT, fs,
                    (0, 160, 255), 1, cv2.LINE_AA)
        y += 20
    if brush_mode:
        cv2.putText(header, f"ANOTACION MANUAL (pincel {brush_radius}px)", (margin, y),
                    FONT, fs, (0, 215, 255), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(header, "clic=marcar der=borrar [ ]=pincel g=guardar", (margin, y),
                    FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
    cv2.putText(header, f"Hallazgos registrados: {n_findings}", (margin, y), FONT, fs,
                (255, 210, 120), 1, cv2.LINE_AA)
    cv2.line(header, (0, height - 1), (width, height - 1), (80, 80, 80), 1)
    return header


def draw_history_section(width, height, findings, scroll_offset):
    section = np.full((height, width, 3), 20, dtype=np.uint8)
    margin = 14
    cv2.putText(section, "HISTORIAL DE HALLAZGOS", (margin, 24), FONT, 0.52,
                (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(section, (margin, 32), (width - margin, 32), (80, 80, 80), 1)
    list_top = 42

    if not findings:
        cv2.putText(section, "Aun no hay hallazgos (automaticos ni manuales).", (margin, list_top + 20),
                    FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        return section, [], 0

    total_content_h = len(findings) * ITEM_H
    rects = []
    y = list_top - scroll_offset
    for item in findings:
        if y + ITEM_H >= list_top and y <= height:
            paste(section, item["thumb"], margin, int(y))
            tag = "[M]" if item.get("tipo") == "manual" else "[A]"
            label = f"{tag} #{item['idx']}  t={seconds_to_mmss(item['time_s'])}"
            tag_col = (0, 215, 255) if item.get("tipo") == "manual" else (120, 255, 120)
            ly = int(y) + THUMB_H + 18
            if list_top <= ly <= height:
                cv2.putText(section, label, (margin, ly), FONT, 0.42, tag_col, 1, cv2.LINE_AA)
        rects.append((int(y), int(y) + ITEM_H, item["idx"]))
        y += ITEM_H

    return section, rects, total_content_h


def draw_button_bar(width, paused, analysis_on=True, render_mode="mask", brush_mode=False):
    bar = np.full((BAR_H, width, 3), 35, dtype=np.uint8)
    n = len(BUTTONS_SPEC)
    bw = width // n
    rects = []
    for i, (label, action) in enumerate(BUTTONS_SPEC):
        x0 = i * bw
        x1 = width - 1 if i == n - 1 else (i + 1) * bw - 4
        cv2.rectangle(bar, (x0 + 4, 8), (x1, BAR_H - 8), (70, 70, 70), -1)
        cv2.rectangle(bar, (x0 + 4, 8), (x1, BAR_H - 8), (120, 120, 120), 1)

        text = label
        color = (235, 235, 235)
        if action == "toggle":
            text = "PLAY" if paused else "PAUSA"
            color = (0, 220, 255)
        elif action == "toggle_analysis":
            text = "ANALISIS: ON" if analysis_on else "ANALISIS: OFF"
            color = (120, 255, 120) if analysis_on else (0, 160, 255)
        elif action == "toggle_mode":
            text = "MODO: MASCARA" if render_mode == "mask" else "MODO: BBOX"
            color = (255, 210, 120)
        elif action == "toggle_manual":
            text = "ANOTAR: ON" if brush_mode else "ANOTAR: OFF"
            color = (0, 215, 255) if brush_mode else (235, 235, 235)
        elif action == "save_manual":
            color = (0, 215, 255)
        elif action == "quit":
            color = (110, 110, 255)

        avail = (x1 - x0 - 4) - 10
        fs = 0.7
        (tw, th), _ = cv2.getTextSize(text, FONT, fs, 2)
        if tw > avail:
            fs *= avail / tw
            fs = max(0.34, fs)
            (tw, th), _ = cv2.getTextSize(text, FONT, fs, 2)
        cv2.putText(bar, text, (x0 + 4 + max(0, ((x1 - x0 - 4) - tw) // 2), (BAR_H + th) // 2),
                    FONT, fs, color, 2, cv2.LINE_AA)
        rects.append((x0 + 4, x1, action))
    return bar, rects


# =========================================================
# 9. SEEK SECUENCIAL
# =========================================================
def seek_sequential(cap, target_frame):
    if target_frame <= 0:
        return
    print(f"Avanzando secuencialmente hasta el frame {target_frame} "
          f"(mas lento pero confiable en .avi/MJPEG)...")
    for i in range(target_frame):
        if not cap.grab():
            print(f"Advertencia: el video se corto en el frame {i}.")
            break
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{target_frame}")


def screen_size(fallback=(1600, 900)):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return fallback


# =========================================================
# 10. REPRODUCTOR PRINCIPAL (multi-video + anotacion manual)
# =========================================================
def run_viewer(source, checkpoint_path, output_path=None, size=352,
               device=None, show_display=True, threshold=0.5, max_frames=None,
               start_frame=None, end_frame=None, cache_frames=150,
               display_width=None, display_height=None, show_panel=True,
               video_height=None, panel_width=PANEL_W, history_dir=None,
               playlist=None):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if display_width is None or display_height is None:
        scr_w, scr_h = screen_size()
        display_width = display_width or scr_w - 40
        display_height = display_height or scr_h - 160

    model = load_model(checkpoint_path, device)

    videos = list(playlist) if playlist else [source]
    try:
        video_idx = videos.index(source)
    except ValueError:
        video_idx = 0

    window_name = "Original | Segmentacion UACANet-v2"
    display_ok = show_display
    window_created = False
    trackbars_created = False

    # preferencias de UI que se conservan al cambiar de video
    threshold = threshold
    analysis_on = True
    render_mode = "mask"
    brush_mode = False
    brush_radius = 15

    # Estos dos los usan los callbacks de los trackbars, que en Windows SOLO
    # se pueden crear una vez por ventana (ver mas abajo el porque). Por eso
    # viven aqui, fuera del loop de videos, en vez de recrearse cada vez que
    # se cambia de video: el callback que OpenCV ya registro sigue
    # apuntando a este mismo diccionario pase lo que pase.
    pending_seek = {"target": None}
    suppress_seek_cb = {"flag": False}

    # Rango del slider de posicion: se usa el video MAS LARGO de la lista
    # para que el mismo slider (creado una sola vez) sirva para cualquiera.
    # Si cambia de video, un numero mas alto que la duracion de ese video en
    # particular simplemente no hace nada raro (jump_to se detiene al
    # llegar al final).
    max_total_frames = 0
    for v in videos:
        if isinstance(v, int) or str(v).isdigit():
            continue
        probe = cv2.VideoCapture(v)
        if probe.isOpened():
            max_total_frames = max(max_total_frames, int(probe.get(cv2.CAP_PROP_FRAME_COUNT)))
        probe.release()

    print("\nControles: ESPACIO=pausa | s=stop | a/d=+-1 frame | z/x=+-10 | -/+=umbral | "
          "v=analisis on/off | m=modo mascara/bbox | e=anotar manual | [ ]=pincel | "
          "c=borrar marca | g=guardar marca | n/p=siguiente/anterior video | q=salir\n")

    while True:
        cur_source = videos[video_idx]
        cap_source = int(cur_source) if str(cur_source).isdigit() else cur_source
        is_camera = isinstance(cap_source, int)

        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"No se pudo abrir: {cur_source}; se omite.")
            if len(videos) == 1:
                raise RuntimeError(f"No se pudo abrir la fuente de video: {cur_source}")
            video_idx = (video_idx + 1) % len(videos)
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cur_history_dir = history_dir if (history_dir and len(videos) == 1) else default_history_dir(cur_source)
        findings, manual_annotations = load_history(cur_history_dir, src_fps)
        registered_frames = {f["idx"] for f in findings if f.get("tipo", "automatico") == "automatico"}
        if findings:
            print(f"Se cargaron {len(findings)} hallazgo(s) previos de '{cur_history_dir}'.")

        # --start-time/--end-time/--max-frames solo aplican al primer video de la lista;
        # al cambiar de video (n/p) cada uno se procesa completo desde el inicio.
        start_frame_eff = (start_frame or 0) if video_idx == 0 else 0
        if start_frame_eff:
            seek_sequential(cap, start_frame_eff)

        max_frames_eff = max_frames if video_idx == 0 else None
        if video_idx == 0 and end_frame is not None:
            remaining = end_frame - start_frame_eff
            max_frames_eff = remaining if max_frames_eff is None else min(max_frames_eff, remaining)

        panel_total = (SEP_W + panel_width) if show_panel else 0
        max_video_w = max(200, (display_width - SEP_W - panel_total) // 2)
        max_video_h = max(200, display_height - BAR_H)
        if video_height is not None:
            scale = video_height / frame_h
        else:
            scale = min(max_video_w / frame_w, max_video_h / frame_h)
        scale = min(scale, max_video_w / frame_w, max_video_h / frame_h)
        disp_w = max(240, int(round(frame_w * scale)))
        disp_h = max(200, int(round(frame_h * scale)))
        canvas_w = disp_w * 2 + SEP_W + panel_total
        min_panel_h = 560 if show_panel else 0
        canvas_h = min(max_video_h, max(disp_h, min_panel_h))
        pad_top = (canvas_h - disp_h) // 2
        panel_x0 = disp_w * 2 + SEP_W * 2

        video_label = "Camara en vivo" if is_camera else \
            f"Video {video_idx + 1}/{len(videos)}: {os.path.basename(str(cur_source))}"
        print(f"\n{video_label}")
        print(f"{frame_w}x{frame_h} -> panel {disp_w}x{disp_h} | ventana {canvas_w}x{canvas_h + BAR_H}")
        print(f"Historial: {cur_history_dir}")

        win_scale = min(1.0, display_width / canvas_w) if canvas_w else 1.0
        win_w = max(1, int(canvas_w * win_scale))
        win_h = max(1, int((canvas_h + BAR_H) * win_scale))

        if display_ok and not window_created:
            try:
                # WINDOW_NORMAL (no AUTOSIZE) a proposito: con trackbars en la
                # ventana, dejar que una AUTOSIZE se auto-redimensione cuando
                # llega una imagen de OTRO tamano (justo lo que pasa al
                # cambiar de video, porque cada uno puede tener resolucion
                # distinta) es un crash nativo conocido de OpenCV en Windows.
                # Con NORMAL controlamos el tamano nosotros via resizeWindow.
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True
            except cv2.error:
                print("No hay backend grafico disponible (ej. Kaggle). Desactivando ventana en vivo.")
                display_ok = False

        if display_ok and window_created:
            # Se redimensiona EXPLICITAMENTE cada vez que cambia el video
            # (en vez de dejar que imshow lo infiera solo), porque es justo
            # esa auto-resize implicita con trackbars puestos la que crashea.
            cv2.resizeWindow(window_name, win_w, win_h)

        cur_output_path = output_path
        if output_path and len(videos) > 1:
            stem, ext = os.path.splitext(output_path)
            cur_output_path = f"{stem}_{os.path.splitext(os.path.basename(str(cur_source)))[0]}{ext or '.mp4'}"

        writer = None
        if cur_output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(cur_output_path, fourcc, src_fps, (canvas_w, canvas_h))
            print(f"Guardando salida en: {cur_output_path}")

        # --- estado de esta sesion de video ---
        cache = deque(maxlen=max(2, cache_frames))
        pos = -1
        paused = False
        read_idx = start_frame_eff
        processed = 0
        failed_reads = 0
        pending_step = 0
        pending_seek["target"] = None           # reset logico; el dict en si se conserva entre videos
        scale_state = {"s": 1.0}
        rects_state = {"r": []}
        scroll_offset = 0
        max_scroll = 0
        history_rects = []
        is_drawing = False
        erase_mode = False
        nav = {"action": None}   # None | "quit_app" | "next_video" | "prev_video"
        t_start = time.time()

        def read_and_process():
            nonlocal pos, read_idx, processed, failed_reads
            consecutive = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    break
                failed_reads += 1
                consecutive += 1
                read_idx += 1
                if consecutive >= 30:
                    return False

            if analysis_on:
                mask_prob, _ = segment_frame(model, frame, device, size=size)
                prob_u8 = (mask_prob * 255).astype(np.uint8)
            else:
                prob_u8 = None

            cache.append({"frame": frame, "prob": prob_u8, "idx": read_idx, "written": False})
            pos = len(cache) - 1
            read_idx += 1
            processed += 1
            return True

        def jump_to(target):
            nonlocal pos, read_idx, cap
            target = max(0, target)

            for i, e in enumerate(cache):
                if e["idx"] == target:
                    pos = i
                    return
            if cache and target > cache[-1]["idx"]:
                to_skip = target - read_idx
                if to_skip > 0:
                    print(f"Saltando {to_skip} frames hasta el {target} (sin decodificar)...")
                    for _ in range(to_skip):
                        if not cap.grab():
                            break
                        read_idx += 1
                read_and_process()
                return

            if is_camera:
                print("No se puede retroceder en una camara en vivo; se ignora el salto.")
                return
            print(f"Reabriendo el video para llegar al frame {target} (salto hacia atras)...")
            cap.release()
            cap = cv2.VideoCapture(cap_source)
            seek_sequential(cap, target)
            cache.clear()
            read_idx = target
            read_and_process()

        def locate_overlay_coords(xc, yc):
            """Convierte una coordenada de ventana (ya des-escalada) a coordenadas
            del frame original, solo si cae dentro del panel de SEGMENTACION."""
            if not (pad_top <= yc < pad_top + disp_h):
                return None
            x0 = disp_w + SEP_W
            if not (x0 <= xc < x0 + disp_w):
                return None
            return (xc - x0) / scale, (yc - pad_top) / scale

        def paint_at(fx, fy, erase):
            if pos < 0:
                return
            entry = cache[pos]
            idx = entry["idx"]
            if idx not in manual_annotations:
                manual_annotations[idx] = np.zeros(entry["frame"].shape[:2], dtype=np.uint8)
            cv2.circle(manual_annotations[idx], (int(fx), int(fy)),
                      brush_radius, 0 if erase else 255, -1)

        def save_manual_mark():
            if pos < 0:
                return
            entry = cache[pos]
            idx = entry["idx"]
            mask = manual_annotations.get(idx)
            if mask is None or not mask.any():
                print("No hay marca manual en este frame para guardar.")
                return
            ys, xs = np.where(mask > 0)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            temp = entry["frame"].copy()
            color_layer = np.zeros_like(temp)
            color_layer[:] = (0, 215, 255)
            m3 = (mask > 0)[:, :, None]
            temp = np.where(m3, cv2.addWeighted(temp, 0.55, color_layer, 0.45, 0), temp)
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(temp, contours, -1, (0, 165, 255), 2)
            thumb = make_finding_thumbnail(temp, bbox, THUMB_W, THUMB_H)
            findings[:] = [it for it in findings if not (it["idx"] == idx and it.get("tipo") == "manual")]
            findings.append({"idx": idx, "time_s": idx / src_fps if src_fps else 0.0,
                             "thumb": thumb, "tipo": "manual", "mask": mask.copy()})
            registered_frames.add(idx)
            print(f"Marca manual guardada (frame {idx}).")

        def on_mouse(event, x, y, flags, param):
            nonlocal paused, pending_step, threshold, analysis_on, render_mode, \
                scroll_offset, brush_mode, is_drawing, erase_mode
            s = scale_state["s"]
            xc, yc = x / s, y / s

            if event == cv2.EVENT_LBUTTONDOWN:
                if yc < canvas_h:
                    if show_panel and xc >= panel_x0:
                        for ry0, ry1, fidx in history_rects:
                            if ry0 <= (yc - HEADER_H) <= ry1:
                                paused = True
                                pending_seek["target"] = fidx
                                break
                    elif brush_mode:
                        coords = locate_overlay_coords(xc, yc)
                        if coords:
                            is_drawing = True
                            erase_mode = False
                            paint_at(*coords, erase=False)
                    return
                for x0, x1, action in rects_state["r"]:
                    if x0 <= xc <= x1:
                        if action == "toggle":
                            paused = not paused
                        elif action == "stop":
                            paused = True
                            pending_seek["target"] = 0
                        elif action == "back1":
                            pending_step = -1
                        elif action == "back10":
                            pending_step = -10
                        elif action == "fwd1":
                            pending_step = 1
                        elif action == "fwd10":
                            pending_step = 10
                        elif action == "thr_down":
                            threshold = max(0.05, threshold - 0.05)
                        elif action == "thr_up":
                            threshold = min(0.95, threshold + 0.05)
                        elif action == "toggle_analysis":
                            analysis_on = not analysis_on
                        elif action == "toggle_mode":
                            render_mode = "bbox" if render_mode == "mask" else "mask"
                        elif action == "toggle_manual":
                            brush_mode = not brush_mode
                            if brush_mode:
                                paused = True
                        elif action == "save_manual":
                            save_manual_mark()
                        elif action == "prev_video":
                            nav["action"] = "prev_video"
                        elif action == "next_video":
                            nav["action"] = "next_video"
                        elif action == "quit":
                            nav["action"] = "quit_app"
                        break

            elif event == cv2.EVENT_RBUTTONDOWN:
                if yc < canvas_h and brush_mode and not (show_panel and xc >= panel_x0):
                    coords = locate_overlay_coords(xc, yc)
                    if coords:
                        is_drawing = True
                        erase_mode = True
                        paint_at(*coords, erase=True)

            elif event == cv2.EVENT_MOUSEMOVE:
                if is_drawing:
                    coords = locate_overlay_coords(xc, yc)
                    if coords:
                        paint_at(*coords, erase=erase_mode)

            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                is_drawing = False

            elif event == cv2.EVENT_MOUSEWHEEL:
                if yc < canvas_h and show_panel and xc >= panel_x0 and (yc - HEADER_H) >= 0:
                    wheel = np.int16(flags >> 16)
                    step = 45
                    if wheel > 0:
                        scroll_offset = max(0, scroll_offset - step)
                    else:
                        scroll_offset = min(max_scroll, scroll_offset + step)

        if display_ok:
            # setMouseCallback SI se puede volver a llamar en cada video sin
            # problema (solo reemplaza un puntero a funcion); lo que NO es
            # seguro en Windows es recrear los TRACKBARS cada vez -> por eso
            # estos se crean una unica vez, la primera pasada por aqui.
            cv2.setMouseCallback(window_name, on_mouse)

            if not trackbars_created:
                def _on_threshold_trackbar(val):
                    nonlocal threshold
                    threshold = max(0.05, val / 100.0)

                def _on_position_trackbar(val):
                    if suppress_seek_cb["flag"]:
                        return
                    pending_seek["target"] = val

                cv2.createTrackbar("Umbral x100", window_name, int(threshold * 100), 95,
                                   _on_threshold_trackbar)
                if max_total_frames > 0:
                    cv2.createTrackbar("Posicion", window_name, start_frame_eff,
                                       max(1, max_total_frames - 1), _on_position_trackbar)
                else:
                    print("Aviso: no se pudo determinar la duracion de los videos; "
                          "el slider de posicion queda desactivado (usa los botones).")
                trackbars_created = True
            else:
                # la ventana y los trackbars ya existian de un video anterior;
                # solo se refleja el umbral vigente, nada se vuelve a crear.
                cv2.setTrackbarPos("Umbral x100", window_name, int(round(threshold * 100)))

        def build_canvas(entry, prev_entry):
            nonlocal max_scroll, history_rects

            if entry["prob"] is None and analysis_on:
                mask_prob, _ = segment_frame(model, entry["frame"], device, size=size)
                entry["prob"] = (mask_prob * 255).astype(np.uint8)

            if entry["prob"] is None:
                overlay = entry["frame"].copy()
            else:
                mask_prob = entry["prob"].astype(np.float32) / 255.0
                overlay, binary = make_overlay(entry["frame"], mask_prob, threshold=threshold,
                                               mode=render_mode)
                region_info = analyze_regions(binary)
                has_detection = region_info["has_detection"]

                prev_had_detection = False
                if prev_entry is not None and prev_entry["prob"] is not None:
                    prev_binary = ((prev_entry["prob"].astype(np.float32) / 255.0) > threshold).astype(np.uint8)
                    prev_had_detection = analyze_regions(prev_binary)["has_detection"]

                if has_detection and not prev_had_detection and entry["idx"] not in registered_frames:
                    thumb = make_finding_thumbnail(overlay, region_info["bbox"], THUMB_W, THUMB_H)
                    findings.append({"idx": entry["idx"], "time_s": entry["idx"] / src_fps if src_fps else 0.0,
                                     "thumb": thumb, "tipo": "automatico"})
                    registered_frames.add(entry["idx"])

            manual_mask = manual_annotations.get(entry["idx"])
            if manual_mask is not None and manual_mask.any():
                color_layer = np.zeros_like(overlay)
                color_layer[:] = (0, 215, 255)
                m3 = (manual_mask > 0)[:, :, None]
                overlay = np.where(m3, cv2.addWeighted(overlay, 0.55, color_layer, 0.45, 0), overlay)
                contours, _ = cv2.findContours((manual_mask > 0).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 165, 255), 2)

            interp = cv2.INTER_AREA if disp_w < frame_w else cv2.INTER_LINEAR
            vid = cv2.resize(entry["frame"], (disp_w, disp_h), interpolation=interp)
            ovr = cv2.resize(overlay, (disp_w, disp_h), interpolation=interp)

            def column(img):
                if canvas_h == disp_h:
                    return img
                col = np.zeros((canvas_h, disp_w, 3), dtype=np.uint8)
                col[pad_top:pad_top + disp_h] = img
                return col

            sep = np.full((canvas_h, SEP_W, 3), 40, dtype=np.uint8)
            parts = [column(vid), sep, column(ovr)]
            if show_panel:
                t_s = entry["idx"] / src_fps if src_fps else 0.0
                header = draw_status_header(panel_width, HEADER_H, entry["idx"], t_s, paused,
                                            threshold, render_mode, analysis_on, len(findings),
                                            brush_mode, brush_radius, video_label)
                hist_h = canvas_h - HEADER_H
                hist_section, history_rects, total_content_h = draw_history_section(
                    panel_width, hist_h, findings, scroll_offset)
                max_scroll = max(0, total_content_h - hist_h)
                panel = np.vstack([header, hist_section])
                parts += [sep.copy(), panel]
            canvas = np.hstack(parts)

            ty = pad_top + 28
            label_right = "Segmentacion UACANet-v2" if analysis_on else "Segmentacion (analisis OFF)"
            right_col = (255, 255, 255) if analysis_on else (0, 165, 255)
            if brush_mode:
                label_right += "  [ANOTACION MANUAL]"
                right_col = (0, 215, 255)
            cv2.putText(canvas, "Video original", (10, ty), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, label_right, (disp_w + SEP_W + 10, ty), FONT, 0.8, right_col, 2, cv2.LINE_AA)
            return canvas

        try:
            while nav["action"] is None:
                if pending_seek["target"] is not None:
                    target = pending_seek["target"]
                    pending_seek["target"] = None
                    paused = True
                    jump_to(target)

                if pending_step != 0:
                    paused = True
                    if pending_step < 0:
                        pos = max(0, pos + pending_step)
                    else:
                        for _ in range(pending_step):
                            if pos < len(cache) - 1:
                                pos += 1
                            elif not read_and_process():
                                break
                    pending_step = 0
                elif not paused:
                    if pos < len(cache) - 1:
                        pos += 1
                    else:
                        if max_frames_eff is not None and processed >= max_frames_eff:
                            print("\nSe alcanzo el limite de frames solicitado.")
                            paused = True
                            if not display_ok:
                                nav["action"] = "quit_app"
                                break
                        elif not read_and_process():
                            print("\nFin del video (o lecturas fallidas seguidas).")
                            paused = True
                            if not display_ok:
                                nav["action"] = "quit_app"
                                break

                if pos < 0:
                    if not read_and_process():
                        nav["action"] = "quit_app"
                        break

                entry = cache[pos]
                prev_entry = cache[pos - 1] if pos > 0 else None
                canvas = build_canvas(entry, prev_entry)

                if writer is not None and not entry["written"]:
                    writer.write(canvas[:canvas_h, :canvas_w])
                    entry["written"] = True

                if not display_ok:
                    continue

                bar, rects = draw_button_bar(canvas_w, paused, analysis_on, render_mode, brush_mode)
                rects_state["r"] = rects
                full = np.vstack([canvas, bar])

                s = min(1.0, display_width / full.shape[1])
                scale_state["s"] = s
                shown = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else full
                cv2.imshow(window_name, shown)

                cv2.setTrackbarPos("Umbral x100", window_name, int(round(threshold * 100)))
                if max_total_frames > 0:
                    suppress_seek_cb["flag"] = True
                    cv2.setTrackbarPos("Posicion", window_name, min(entry["idx"], max_total_frames - 1))
                    suppress_seek_cb["flag"] = False

                key = cv2.waitKey(30 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    nav["action"] = "quit_app"
                elif key == 32:
                    paused = not paused
                elif key == ord("s"):
                    paused = True
                    pending_seek["target"] = 0
                elif key in (ord("a"), 81):
                    pending_step = -1
                elif key in (ord("d"), 83):
                    pending_step = 1
                elif key == ord("z"):
                    pending_step = -10
                elif key == ord("x"):
                    pending_step = 10
                elif key in (ord("-"), ord("_")):
                    threshold = max(0.05, threshold - 0.05)
                elif key in (ord("+"), ord("=")):
                    threshold = min(0.95, threshold + 0.05)
                elif key == ord("v"):
                    analysis_on = not analysis_on
                elif key == ord("m"):
                    render_mode = "bbox" if render_mode == "mask" else "mask"
                elif key == ord("e"):
                    brush_mode = not brush_mode
                    if brush_mode:
                        paused = True
                elif key == ord("["):
                    brush_radius = max(3, brush_radius - 3)
                elif key == ord("]"):
                    brush_radius = min(60, brush_radius + 3)
                elif key == ord("c"):
                    if pos >= 0:
                        manual_annotations.pop(cache[pos]["idx"], None)
                elif key == ord("g"):
                    save_manual_mark()
                elif key == ord("n") and len(videos) > 1:
                    nav["action"] = "next_video"
                elif key == ord("p") and len(videos) > 1:
                    nav["action"] = "prev_video"
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            save_history(findings, cur_history_dir)

        elapsed = time.time() - t_start
        print(f"\nProcesados {processed} frames en {elapsed:.1f}s")
        if failed_reads:
            print(f"Nota: {failed_reads} lecturas fallaron (cuadros corruptos del .avi) y se saltaron.")

        if nav["action"] in (None, "quit_app"):
            break
        elif nav["action"] == "next_video":
            video_idx = (video_idx + 1) % len(videos)
        else:
            video_idx = (video_idx - 1) % len(videos)

    if display_ok:
        cv2.destroyAllWindows()


# =========================================================
# 11. UN SOLO FRAME
# =========================================================
def process_single_frame(source, checkpoint_path, frame_index, output_path=None,
                         size=352, device=None, threshold=0.5, show_display=True,
                         render_mode="mask"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(checkpoint_path, device)

    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and frame_index >= total_frames:
        raise ValueError(f"El video tiene {total_frames} frames; pediste el {frame_index}")

    seek_sequential(cap, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"No se pudo leer el frame {frame_index}; prueba un indice cercano.")

    mask_prob, _ = segment_frame(model, frame, device, size=size)
    overlay, binary = make_overlay(frame, mask_prob, threshold=threshold, mode=render_mode)
    has_detection = analyze_regions(binary)["has_detection"]

    sep = np.full((frame.shape[0], SEP_W, 3), 40, dtype=np.uint8)
    canvas = np.hstack([frame, sep, overlay])

    cv2.putText(canvas, f"Frame {frame_index} - original", (10, 25), FONT, 0.7, (255, 255, 255), 2)
    label = "Segmentacion UACANet-v2" + (" - POLIPO DETECTADO" if has_detection else "")
    color = (0, 0, 255) if has_detection else (255, 255, 255)
    cv2.putText(canvas, label, (frame.shape[1] + SEP_W + 10, 25), FONT, 0.7, color, 2)

    if output_path is None:
        output_path = f"frame_{frame_index:06d}_comparacion.png"
    cv2.imwrite(output_path, canvas)
    print(f"Guardado: {output_path}")

    if has_detection:
        thumb_dir = default_history_dir(source)
        os.makedirs(thumb_dir, exist_ok=True)
        thumb_path = os.path.join(thumb_dir, f"hallazgo_frame{frame_index:06d}.png")
        cv2.imwrite(thumb_path, overlay)
        print(f"Polipo detectado: miniatura guardada en {thumb_path}")

    if show_display:
        try:
            cv2.imshow(f"Frame {frame_index}", canvas)
            print("Presiona cualquier tecla en la ventana para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("No hay backend grafico disponible; se guardo solo la imagen.")

    return output_path


# =========================================================
# 12. MODO INTERACTIVO + SELECCION DE VIDEO
# =========================================================
def prompt_time_range(source):
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps else 0
    cap.release()

    print(f"\nVideo: {source}")
    print(f"  Duracion: {seconds_to_mmss(duration_s)}  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print("Ingresa el lapso a procesar (mm:ss). Enter vacio = valor por defecto.\n")

    start_str = input("  Inicio [0:00]: ").strip() or "0:00"
    end_str = input(f"  Fin [{seconds_to_mmss(duration_s)}]: ").strip() or seconds_to_mmss(duration_s)

    start_f = parse_timecode(start_str, fps)
    end_f = parse_timecode(end_str, fps)
    if start_f < 0 or start_f >= end_f:
        raise ValueError(f"Rango invalido: {start_str} -> {start_f}, {end_str} -> {end_f}")

    print(f"\nProcesando de {start_str} (frame {start_f}) a {end_str} (frame {end_f})\n")
    return start_f, end_f


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}


def list_videos_in_dir(folder):
    files = [f for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    return [os.path.join(folder, f) for f in files]


def prompt_video_choice(videos):
    print("\nVideos disponibles:")
    for i, v in enumerate(videos):
        print(f"  [{i}] {os.path.basename(v)}")
    choice = input(f"Elige un video [0-{len(videos) - 1}] (Enter = 0): ").strip()
    if choice == "":
        return 0
    try:
        idx = int(choice)
        if 0 <= idx < len(videos):
            return idx
    except ValueError:
        pass
    print("Opcion invalida; se usa el primero.")
    return 0


# =========================================================
# 12b. SELECTOR GRAFICO (miniaturas, clic para elegir)
# =========================================================
def pick_video_gui(checkpoint=None, videos=None):
    """
    Ventana con una miniatura por video (el primer frame de cada uno) para
    elegir con un clic, en vez de escribir rutas o numeros en la consola.
    Pensado para que un doctor no tenga que tocar la terminal.

    - Si 'videos' ya viene dado (porque se paso --library), se muestran
      directo, con un boton para cambiar de carpeta si hace falta.
    - Si no viene nada, aparecen botones para elegir una carpeta o un solo
      archivo.
    - Si 'checkpoint' no viene dado, tambien se puede elegir el modelo
      (.pth) desde la misma ventana.

    Devuelve (checkpoint_elegido, lista_de_videos, indice_inicial) o
    (None, None, 0) si se cerro sin elegir nada, o si falta Pillow/tkinter
    (en ese caso avisa por consola y quien llama debe usar el modo texto).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from PIL import Image, ImageTk
    except Exception as e:
        print(f"Aviso: selector grafico no disponible ({e}). "
              f"Instala Pillow ('pip install Pillow') o usa --library con --no-display para el modo texto.")
        return None, None, 0

    THUMB_W, THUMB_H = 220, 140
    COLS = 4

    root = tk.Tk()
    root.title("UACANet - elegir video")
    root.configure(bg="#1e1e1e")
    root.geometry("980x680")

    state = {"checkpoint": checkpoint, "videos": list(videos) if videos else [],
            "start_idx": 0, "confirmed": False}
    thumb_refs = []   # hay que mantener una referencia o Tk las "recoge" y la imagen se borra

    top = tk.Frame(root, bg="#1e1e1e")
    top.pack(fill="x", padx=12, pady=10)

    ckpt_label = tk.Label(top, text="", bg="#1e1e1e", fg="#dddddd", anchor="w",
                          font=("Segoe UI", 10))
    ckpt_label.pack(side="left", fill="x", expand=True)

    def refresh_ckpt_label():
        if state["checkpoint"]:
            ckpt_label.config(text=f"Modelo: {os.path.basename(state['checkpoint'])}")
        else:
            ckpt_label.config(text="Modelo: (sin elegir)")

    def choose_checkpoint():
        path = filedialog.askopenfilename(
            title="Selecciona el modelo entrenado",
            filetypes=[("Checkpoint PyTorch", "*.pth"), ("Todos los archivos", "*.*")])
        if path:
            state["checkpoint"] = path
            refresh_ckpt_label()

    tk.Button(top, text="Elegir modelo (.pth)...", command=choose_checkpoint).pack(side="right")
    refresh_ckpt_label()

    status = tk.Label(root, text="", bg="#1e1e1e", fg="#9fd3ff", font=("Segoe UI", 9))
    status.pack(fill="x", padx=12)

    # --- grilla de miniaturas, con scroll ---
    container = tk.Frame(root, bg="#1e1e1e")
    container.pack(fill="both", expand=True, padx=12, pady=10)
    gcanvas = tk.Canvas(container, bg="#1e1e1e", highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=gcanvas.yview)
    grid_frame = tk.Frame(gcanvas, bg="#1e1e1e")
    grid_frame.bind("<Configure>", lambda e: gcanvas.configure(scrollregion=gcanvas.bbox("all")))
    gcanvas.create_window((0, 0), window=grid_frame, anchor="nw")
    gcanvas.configure(yscrollcommand=scrollbar.set)
    gcanvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_wheel(event):
        gcanvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
    gcanvas.bind_all("<MouseWheel>", _on_wheel)

    def make_thumbnail(path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        canvas_img = np.full((THUMB_H, THUMB_W, 3), 35, dtype=np.uint8)
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = frame.shape[:2]
            scale = min(THUMB_W / fw, THUMB_H / fh)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            x0, y0 = (THUMB_W - nw) // 2, (THUMB_H - nh) // 2
            canvas_img[y0:y0 + nh, x0:x0 + nw] = frame
        return ImageTk.PhotoImage(Image.fromarray(canvas_img))

    def on_pick(idx):
        state["start_idx"] = idx
        state["confirmed"] = True
        root.destroy()

    def populate():
        for w in grid_frame.winfo_children():
            w.destroy()
        thumb_refs.clear()
        vids = state["videos"]
        if not vids:
            tk.Label(grid_frame, text="Elige una carpeta o un video para empezar.",
                    bg="#1e1e1e", fg="#999999", font=("Segoe UI", 11)).grid(row=0, column=0, padx=20, pady=20)
            return
        for i, path in enumerate(vids):
            status.config(text=f"Generando miniatura {i + 1}/{len(vids)}...")
            root.update_idletasks()
            thumb = make_thumbnail(path)
            thumb_refs.append(thumb)
            cell = tk.Frame(grid_frame, bg="#1e1e1e")
            cell.grid(row=i // COLS, column=i % COLS, padx=10, pady=10)
            btn = tk.Button(cell, image=thumb, command=lambda i=i: on_pick(i),
                            bd=2, relief="groove", cursor="hand2")
            btn.pack()
            name = os.path.basename(path)
            if len(name) > 30:
                name = name[:27] + "..."
            tk.Label(cell, text=name, bg="#1e1e1e", fg="#dddddd", font=("Segoe UI", 9)).pack()
        status.config(text=f"{len(vids)} video(s) — clic en uno para abrirlo.")

    def choose_folder():
        folder = filedialog.askdirectory(title="Selecciona la carpeta con los videos")
        if not folder:
            return
        found = list_videos_in_dir(folder)
        if not found:
            messagebox.showwarning("Sin videos", "No se encontraron videos (.avi/.mp4/.mov/...) en esa carpeta.")
            return
        state["videos"] = found
        populate()

    def choose_single_file():
        path = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[("Videos", "*.avi *.mp4 *.mov *.mkv *.mpg *.mpeg *.wmv"), ("Todos", "*.*")])
        if path:
            state["videos"] = [path]
            populate()

    browse = tk.Frame(root, bg="#1e1e1e")
    browse.pack(fill="x", padx=12, pady=(0, 10))
    tk.Button(browse, text="Elegir carpeta de videos...", command=choose_folder).pack(side="left")
    tk.Button(browse, text="Elegir un solo video...", command=choose_single_file).pack(side="left", padx=10)

    populate()
    root.mainloop()

    if not state["confirmed"] or not state["videos"]:
        return None, None, 0
    return state["checkpoint"], state["videos"], state["start_idx"]


# =========================================================
# 13. CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visor con controles: original vs segmentacion UACANet-v2, "
                                                  "con historial automatico+manual y navegacion entre videos")
    parser.add_argument("--source", default=None, help="Ruta a video, o indice de camara (0 = webcam)")
    parser.add_argument("--library", default=None,
                        help="Carpeta con varios videos: al iniciar eliges cual ver, y dentro del visor "
                             "usas n/p (o los botones <VID / VID>) para cambiar de video.")
    parser.add_argument("--checkpoint", default=None,
                        help="Ruta a best_model.pth / last_model.pth. Si se omite y se usa el selector "
                             "grafico, se puede elegir ahi mismo.")
    parser.add_argument("--no-picker", action="store_true",
                        help="No abrir el selector grafico aunque falten --source/--library; "
                             "usar el modo de consola (numero + Enter) en su lugar.")
    parser.add_argument("--output", default=None, help="Ruta .mp4 para guardar el resultado (opcional)")
    parser.add_argument("--history-dir", default=None,
                        help="Carpeta donde guardar el historial de hallazgos (imagenes + CSV). "
                             "Solo aplica con un unico video; con --library se usa una carpeta por video.")
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true", help="No abrir ventana (Kaggle/servidor)")
    parser.add_argument("--no-panel", action="store_true", help="Ocultar el panel lateral (estado + historial)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cache-frames", type=int, default=150,
                        help="Cuantos frames guardar en memoria para poder retroceder")
    parser.add_argument("--display-width", type=int, default=None)
    parser.add_argument("--display-height", type=int, default=None)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=PANEL_W)
    parser.add_argument("--frame", type=int, default=None, help="Procesa UN SOLO frame por indice")
    parser.add_argument("--time", type=str, default=None, help="Igual que --frame pero por timecode ('7:16')")
    parser.add_argument("--start-time", type=str, default=None)
    parser.add_argument("--end-time", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    videos = None
    if args.library:
        videos = list_videos_in_dir(args.library)
        if not videos:
            raise SystemExit(f"No se encontraron videos en {args.library}")
        start_idx = 0
        if not args.no_display and len(videos) > 1:
            if args.no_picker:
                start_idx = prompt_video_choice(videos)
            else:
                # el selector grafico ya recibe la carpeta cargada: solo hay
                # que elegir cual abrir (o cambiar de carpeta ahi mismo)
                ckpt_gui, videos_gui, idx_gui = pick_video_gui(args.checkpoint, videos)
                if videos_gui:
                    videos = videos_gui
                    start_idx = idx_gui
                    if ckpt_gui and not args.checkpoint:
                        args.checkpoint = ckpt_gui
                else:
                    raise SystemExit("No se eligio ningun video; se cierra el programa.")
        chosen_source = videos[start_idx]
    elif args.source:
        chosen_source = args.source
    else:
        # Ni --source ni --library: se abre el selector grafico desde cero
        # (si --no-display o falta Pillow/tkinter, se cae al modo consola).
        if args.no_display or args.no_picker:
            parser.error("Indica --source (un video o camara) o --library (una carpeta con varios videos). "
                         "El selector grafico no aplica con --no-display/--no-picker.")
        ckpt_gui, videos_gui, idx_gui = pick_video_gui(args.checkpoint)
        if not videos_gui:
            raise SystemExit("No se eligio ningun video; se cierra el programa.")
        if ckpt_gui and not args.checkpoint:
            args.checkpoint = ckpt_gui
        videos = videos_gui if len(videos_gui) > 1 else None
        chosen_source = videos_gui[idx_gui]

    if not args.checkpoint:
        parser.error("Falta el modelo: pasa --checkpoint o eligelo en el selector grafico.")

    def _get_fps(src):
        cap_probe = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        cap_probe.release()
        return fps

    frame_index = args.frame
    if args.time is not None:
        fps = _get_fps(chosen_source)
        frame_index = parse_timecode(args.time, fps)
        print(f"Timecode {args.time} -> frame {frame_index} (fps detectado: {fps:.2f})")

    common = dict(
        source=chosen_source, checkpoint_path=args.checkpoint, output_path=args.output,
        size=args.size, show_display=not args.no_display, threshold=args.threshold,
        max_frames=args.max_frames, cache_frames=args.cache_frames,
        display_width=args.display_width, display_height=args.display_height,
        show_panel=not args.no_panel,
        video_height=args.video_height, panel_width=args.panel_width,
        history_dir=args.history_dir, playlist=videos,
    )

    if args.interactive:
        start_f, end_f = prompt_time_range(chosen_source)
        run_viewer(start_frame=start_f, end_frame=end_f, **common)
    elif frame_index is not None:
        process_single_frame(
            source=chosen_source, checkpoint_path=args.checkpoint, frame_index=frame_index,
            output_path=args.output, size=args.size, threshold=args.threshold,
            show_display=not args.no_display,
        )
    elif args.start_time is not None:
        fps = _get_fps(chosen_source)
        start_f = parse_timecode(args.start_time, fps)
        end_f = parse_timecode(args.end_time, fps) if args.end_time else None
        print(f"Rango: frame {start_f}" + (f" -> {end_f}" if end_f else ""))
        run_viewer(start_frame=start_f, end_frame=end_f, **common)
    else:
        run_viewer(**common)