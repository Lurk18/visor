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
