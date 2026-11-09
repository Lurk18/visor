# UACANet web

Copia tu `viewer_uacanet.py` en esta carpeta (el servidor lo importa) y:

    pip install -r requirements.txt

El servidor necesita **FFmpeg** instalado (incluyendo `libx264`) para preparar automaticamente los videos subidos en MP4/H.264/`yuv420p`. En Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`. En Windows, instala FFmpeg y agrega `ffmpeg.exe` al `PATH`.
    CHECKPOINT=best_model.pth uvicorn server:app --host 0.0.0.0 --port 8000

Docker (GPU):

    docker build -t uacanet-web .
    docker run --gpus all -p 8000:8000 -v /ruta/modelos:/models uacanet-web

Abre http://localhost:8000. Para usar la cámara desde otro equipo se necesita HTTPS
(pon nginx/Caddy/Traefik delante con certificado).

## Historial y anotación manual
- Los hallazgos (automáticos y marcas manuales) se guardan en `HISTORY_DIR` (por defecto `./historial`),
  una carpeta `<video>_hallazgos` por video, con `hallazgos.csv`, miniaturas y máscaras.
  El formato es el mismo del visor de escritorio. En Docker, monta un volumen: `-v /ruta/historial:/app/historial`.
- Al volver a abrir el mismo video (mismo nombre y tamaño) el historial y las marcas se cargan solos.
- Atajos: `e` anotar, `g` guardar marca, `c` borrar marca, `[` `]` pincel, `a`/`d` ±1 frame, espacio pausa.
  Clic izquierdo pinta, clic derecho borra.
- El número de frame se calcula con 25 fps asumidos (el navegador no expone el fps real).
- **Descartar hallazgos (p. ej. falsos positivos del modelo):** cada hallazgo del panel tiene un botón ✕ (y la barra de nota tiene
  «Descartar hallazgo» cuando hay uno elegido). No se borra nada: el registro queda marcado como `descartado` en
  `predictions/metadata.json` o `annotations/bounding_boxes.json`, con las imágenes en disco (sirven como ejemplos negativos).
  Sale de la lista, de `hallazgos.csv` y de los conteos; la sección «Descartados» permite restaurarlo. Si se descarta uno automático,
  el modelo no vuelve a crear detecciones a ±50 frames (±2 s a 25 fps; ajustable con `DESCARTE_VENTANA_FRAMES`). Las notas ligadas a
  un hallazgo descartado se conservan.
- **Videos guardados (revisión posterior):** la pestaña «Videos guardados» lista todas las colonoscopias de la cuenta (paciente, fecha,
  video, hallazgos, notas y estado de revisión: Pendiente / En revisión / Revisada). «Revisar» abre el video guardado en el visor, con sus
  hallazgos y notas, para analizarlo y anotarlo como siempre; al abrirlo pasa a «En revisión» y «Marcar revisada y volver a la lista»
  lo cierra. El video se reproduce por rangos (`/api/video/<ticket>`, el ticket vence en 8 h), así que se puede avanzar y retroceder sin
  descargarlo entero. Si el original no es reproducible por el navegador (AVI, HEVC, 10 bits...) se convierte una sola vez en el servidor
  (`video/colonoscopia.mp4`, requiere FFmpeg). El estado de revisión se guarda en `metadata/colonoscopy.json` (`revision`).
- **Notas sobre un hallazgo:** haz clic en un hallazgo del panel (o en una nota ya guardada). El video se pausa en ese frame,
  la fila queda resaltada y la nota se guarda ligada a ese hallazgo (`hallazgo: manual|automatico` en `descriptions.json` /
  `transcript.json`). Si mueves el video a otro frame, la selección se quita y la nota vuelve a ir al frame que estás viendo.
- **Finalizar colonoscopia (cámara):** el botón «Finalizar colonoscopia» detiene la cámara, sube el último tramo y pide al servidor
  el cierre. Quedan en la carpeta de la colonoscopia: `video/colonoscopia.mp4` (video completo, navegable; el original que graba el
  navegador se conserva), `notas.csv` y `hallazgos.csv`. Si FFmpeg no está instalado solo queda el original. Si el navegador se cierra
  de golpe se pierde como máximo el último tramo de 5 s, por eso el visor avisa antes de cerrar la pestaña mientras graba.
- **notas.csv:** `frame, tiempo, texto, origen (texto|voz), hallazgo (manual|automatico), hallazgos_en_frame, creado`. UTF-8 con BOM
  (acentos correctos en Excel); se reescribe con cada nota y también se descarga desde el visor. Si Excel lo abre en una sola
  columna, arranca el servidor con `CSV_SEP=";"`.
- **Dictado:** usa el reconocimiento de voz del navegador (Chrome o Edge). Requiere HTTPS (o `http://localhost`), permiso de
  micrófono e internet; no funciona en Firefox ni en Brave. Si falla, el visor muestra el motivo debajo del cuadro de nota.

## Login con Supabase
1. Crea un proyecto en https://supabase.com. En **Project Settings → API** copia la *Project URL* y la clave **anon** (publica).
   No uses nunca la `service_role` en este servicio.
2. En **Authentication → Sign In / Providers → Email**, desactiva *Allow new users to sign up*
   y crea las cuentas de los doctores a mano en **Authentication → Users → Add user**.
3. Crea un archivo `.env` junto a `server.py` (copia `.env.example`) con `SUPABASE_URL` y `SUPABASE_ANON_KEY`.
   **En Windows** la sintaxis `VAR=valor comando` no funciona; usa el `.env` y arranca con
   `uvicorn server:app --port 8000`. Al iniciar, la consola indica si Supabase es alcanzable.
   Si el login muestra *Failed to fetch*, la URL es incorrecta, el proyecto está pausado o la red bloquea Supabase.

   Alternativa en Linux/macOS (variables en la línea de comandos):

       SUPABASE_URL=https://xxxx.supabase.co SUPABASE_ANON_KEY=eyJ... \
       CHECKPOINT=best_model.pth uvicorn server:app --host 0.0.0.0 --port 8000

   En Docker: `-e SUPABASE_URL=... -e SUPABASE_ANON_KEY=...`.

Cada doctor ve solo su propio historial (`historial/<id_usuario>/<video>_hallazgos`).
Supabase solo maneja la identidad: el video y los frames no salen de tu servidor.

## Preparacion automatica de videos para el navegador
- Al seleccionar un video, el navegador lo envia autenticado a `POST /api/video/compatible`.
- FastAPI lo convierte a MP4/H.264, 8 bits, `yuv420p`, sin audio y con `faststart`, usando CRF 18.
- El archivo de entrada y el resultado se manejan en un directorio temporal del servidor; se eliminan despues de enviar la respuesta. El original local del medico no se modifica.
- Limite predeterminado de subida: 2 GiB. Se puede cambiar con `MAX_VIDEO_BYTES` (bytes). Tiempo maximo de conversion: 2 horas.
- La conversion con H.264/CRF 18 es con perdida. Para conservar evidencia clinica, guarda el original fuera del visor y no uses la copia convertida como archivo maestro.
- La conversion consume CPU, disco temporal y ancho de banda; para varios usuarios simultaneos conviene configurar limites y una cola de conversion.
