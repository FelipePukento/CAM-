# CAM-AMFS — Detección de atención grupal (YOLO + Pose)

## TL;DR (uso rápido)
```bash
# 1) Crear venv (opcional) e instalar deps
python -m venv venv && source venv/bin/activate    # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# 2) Colocar los modelos en ./models/
#    - yolo11x.pt
#    - yolo11x-pose.pt

# 3) Ejecutar
python cam-test.py          # pipeline 2-etapas (det -> pose)
# o
python cam-test-PVE.py      # variante con ajustes de tiempo real (si la estás usando)
```

## ¿Qué hace?
- **2 etapas**:  
  1) **Detección de personas** (YOLO 11x – clase 0).  
  2) **Pose por recorte** en cada persona detectada (17 keypoints).  
- Calcula un **score de atención grupal** por frame a partir de features (frontalidad, ángulos de cabeza/torso, manos cerca de la cara, etc.).  
- **ROI** opcional para privilegiar el área de interés.  
- **Detección de teléfonos** (clase 67) para penalizar si está cerca de manos/cara.  
- **CSV opcional** con agregados por frame.

## Requisitos
- Python 3.9+  
- GPU NVIDIA (opcional, recomendado) con drivers/CUDA compatibles para PyTorch.
- Paquetes (ver `requirements.txt`):  
  - `ultralytics`, `torch`/`torchvision` (CUDA si aplica), `opencv-python`, `numpy`…

> Si usas CUDA, instala la build de **torch** acorde a tu versión de CUDA.

## Estructura del repo
```
CAM-IA/
├─ models/                 # colocar aquí: yolo11x.pt, yolo11x-pose.pt
├─ utils/                  # (Modificador de video para mejor rendimiento del modelo)
├─ venv/                   # entorno virtual (opcional)
├─ video/
│  └─ Clase-R_preproc.mp4  # video de prueba / entrada
├─ cam-test.py             # script inicial (2 etapas det->pose)
├─ cam-test-PVE.py         # Script principal con keypoints [MAYOR RENDIMIENTO]
├─ requirements.txt
└─ README.md
```

## Configuración clave (dentro del script)
```python
CONFIG = {
    # Fuente
    "USE_VIDEO_FILE": True,                 # True: archivo / False: cámara
    "VIDEO_PATH": r"video/Clase-R_preproc.mp4",
    "CAM_INDEX": 0,
    "MIRROR": False,

    # Objetivo de tiempo real
    "TARGET_FPS": 30.0,
    "REALTIME_FROM_FILE": True,             # sincroniza playback y salta frames si atrasa

    # Escalado de entrada (lado corto)
    "DOWNSCALE_STEPS": [1080, 720],
    "START_STEP_INDEX": 1,                  # 0..len-1
    "UPSCALE_HEADROOM": 1.15,               # sube escala si sobra FPS
    "DOWNSCALE_PRESSURE": 0.90,             # baja si falta FPS

    # Modelos/umbral
    "MODEL_POSE": "models/yolo11x-pose.pt",
    "MODEL_DET":  "models/yolo11x.pt",
    "CONF_POSE":  0.35,
    "CONF_DET":   0.20,

    # ROI (relativo 0-1): x, y, w, h
    "ROI": [0.28, 0.05, 0.44, 0.65],

    # Ponderaciones y umbrales para el score
    "W_FRONTAL": 15, "W_HEAD": 20, "W_TORSO": 40, "W_HANDS": 25,
    "THR_FRONTAL": 0.25, "THR_HEAD": 30.0, "THR_TORSO": 15.0, "THR_HANDS": 0.50,

    # Penalizaciones/bonos
    "PEN_YAW": 8.0, "PEN_SPEED": 8.0, "BONUS_ROI": 6.0, "PEN_PHONE": 12.0,

    # CSV (si quieres log por frame)
    "CSV_PATH": "",                         # p. ej. "salida.csv"
    "SUMMARY_SECS": 60.0,
}
```

**Notas:**
- **`DOWNSCALE_STEPS`** define resoluciones de entrada progresivas (lado corto). El loop se adapta según FPS para mantener el **TARGET_FPS**.  
- **`ROI`**: si la nariz del esqueleto cae dentro, aplica **BONUS_ROI** (sube el score).  
- **`CONF_DET`/`CONF_POSE`**: bajar un poco aumenta recall (más alumnos detectados) a costa de más falsos positivos y costo.

## Controles y salida en pantalla
- Ventana redimensionable (`ALLOW_RESIZE_WINDOW=True`).  
- Overlay:  
  - **Personas** detectadas  
  - **Score promedio**  
  - Porcentajes **ALTA/MEDIA/BAJA**  
  - Alerta de **teléfono** si se detecta cercano  
  - **FPS** y resolución efectiva (lado corto actual)

CSV (si `CSV_PATH` no está vacío):
```
timestamp, num_personas, score_promedio, %ALTA, %MEDIA, %BAJA, phone_near_group, scale_shorter
```

## Consejos de rendimiento / recall
- Si **falta FPS**:  
  - Aumenta `START_STEP_INDEX` (empieza más pequeño),  
  - Sube `DOWNSCALE_PRESSURE` (baja antes),  
  - O reduce `CONF_DET`/`CONF_POSE` y/o el número de personas procesadas (si usas la variante con límites).
- Si **no detecta suficientes alumnos lejanos**:  
  - Baja `CONF_DET` a ~0.18,  
  - Asegúrate de usar el **modelo grande** (`yolo11x*`) y buen **video fuente** (nitidez/contraste).

## Problemas comunes
- **No abre el video**: revisa `VIDEO_PATH` absoluto y permisos.  
- **CUDA no usado**: verifica instalación de `torch` con CUDA (`torch.cuda.is_available()` en `True`).  
- **Se “cae” el tiempo real**: usa `cam-test-PVE.py` (si la tienes) o reduce `DOWNSCALE_STEPS`/`CONF_*`.

## Licencia
Proyecto académico/experimental. Úsalo bajo tu responsabilidad.
