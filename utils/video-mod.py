import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# =========================
# CONFIG (editar aquí)
# =========================
CONFIG = {
    # Rutas
    "INPUT": r"video/Clase-R.mp4",
    "OUTPUT": r"video/Clase-R_preproc.mp4",

    # Reescalado: si OUT_SHORT_SIDE>0 se ajusta el lado corto manteniendo aspecto
    "OUT_SHORT_SIDE": 0,         # p. ej. 1440, 1080, 0 = sin cambio

    # FPS salida (0 = igual que origen)
    "OUT_FPS": 0,

    # Hilos
    "NUM_WORKERS": 8,            # 2–8 según CPU
    "READER_QUEUE_SIZE": 64,     # tope de frames en cola (equilibrio RAM/latencia)

    # Filtros
    "USE_CLAHE": True,
    "CLAHE_CLIP": 2.0,
    "CLAHE_GRID": 8,

    "USE_GAMMA": True,
    "GAMMA": 1.05,               # 1.00–1.15

    "USE_SAT": True,
    "SAT_GAIN": 1.05,            # 1.00–1.15

    "USE_UNSHARP": True,
    "US_BLUR": 3,                # 3 o 5
    "US_AMOUNT": 0.6,            # 0.4–0.8

    "USE_DENOISE": True,
    "DENOISE_METHOD": "bilateral",  # "nlm" | "bilateral"
    # NLM (lento pero bueno)
    "DN_H": 3, "DN_HColor": 3, "DN_TEMPL": 7, "DN_SRCH": 21,
    # Bilateral (rápido)
    "BIL_DIAM": 5, "BIL_SIGMAC": 50, "BIL_SIGMAS": 5,

    # OpenCV threads (internos)
    "OPENCV_THREADS": 0,         # 0=auto, o un entero
    # Codec ("mp4v" es genérico; si tu backend lo soporta, prueba "avc1")
    "FOURCC": "mp4v"
}

# -------------------------
# Utilidades aceleradas
# -------------------------
class PreprocContext:
    def __init__(self, C):
        self.C = C
        # OpenCV optimizaciones
        cv2.setUseOptimized(True)
        if isinstance(C["OPENCV_THREADS"], int):
            cv2.setNumThreads(C["OPENCV_THREADS"])

        # Pre-crear CLAHE
        self.clahe = None
        if C["USE_CLAHE"]:
            self.clahe = cv2.createCLAHE(clipLimit=C["CLAHE_CLIP"],
                                         tileGridSize=(C["CLAHE_GRID"], C["CLAHE_GRID"]))

        # Pre-calcular LUT de gamma
        self.gamma_lut = None
        if C["USE_GAMMA"]:
            g = max(C["GAMMA"], 1e-6)
            inv = 1.0 / g
            table = (np.linspace(0, 1, 256) ** inv) * 255.0
            self.gamma_lut = np.clip(table, 0, 255).astype(np.uint8)

    def resize_keep_aspect(self, img, short_side):
        if short_side <= 0:
            return img, 1.0
        h, w = img.shape[:2]
        s = min(h, w)
        if s == short_side:
            return img, 1.0
        scale = short_side / float(s)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4), scale

    def apply_clahe_bgr(self, img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = self.clahe.apply(y)
        return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    def apply_gamma(self, img):
        return cv2.LUT(img, self.gamma_lut)

    def apply_saturation(self, img, gain):
        # HSV en float solo para S; evita recomputar todo en float grande
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1].astype(np.float32) * gain
        hsv[..., 1] = np.clip(s, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_unsharp(self, img, ksize, amount):
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

    def apply_denoise(self, img):
        C = self.C
        if C["DENOISE_METHOD"] == "nlm":
            return cv2.fastNlMeansDenoisingColored(img, None, C["DN_H"], C["DN_HColor"], C["DN_TEMPL"], C["DN_SRCH"])
        # Bilateral (notar que no preserva tanto detalle como NLM, pero es mucho más rápido)
        return cv2.bilateralFilter(img, C["BIL_DIAM"], C["BIL_SIGMAC"], C["BIL_SIGMAS"])

    def process(self, frame):
        C = self.C

        # Si vamos a REDUCIR tamaño, hazlo primero para acelerar el resto
        resized_first = False
        if C["OUT_SHORT_SIDE"] > 0:
            h, w = frame.shape[:2]
            scale_target = C["OUT_SHORT_SIDE"] / float(min(h, w))
            if scale_target < 1.0:
                frame, _ = self.resize_keep_aspect(frame, C["OUT_SHORT_SIDE"])
                resized_first = True

        if C["USE_CLAHE"]:
            frame = self.apply_clahe_bgr(frame)
        if C["USE_GAMMA"]:
            frame = self.apply_gamma(frame)
        if C["USE_SAT"]:
            frame = self.apply_saturation(frame, C["SAT_GAIN"])
        if C["USE_UNSHARP"]:
            frame = self.apply_unsharp(frame, C["US_BLUR"], C["US_AMOUNT"])
        if C["USE_DENOISE"]:
            frame = self.apply_denoise(frame)

        # Si íbamos a AUMENTAR tamaño (scale>1), hazlo al final para mejorar nitidez
        if C["OUT_SHORT_SIDE"] > 0 and not resized_first:
            frame, _ = self.resize_keep_aspect(frame, C["OUT_SHORT_SIDE"])

        return frame


def main():
    C = CONFIG
    ctx = PreprocContext(C)

    in_path = Path(C["INPUT"]).resolve()
    out_path = Path(C["OUTPUT"]).resolve()

    cap = cv2.VideoCapture(str(in_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {in_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = C["OUT_FPS"] if C["OUT_FPS"] and C["OUT_FPS"] > 0 else in_fps

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # Leer primer frame para determinar W,H de salida
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("El video no tiene frames")

    proc0 = ctx.process(frame0)
    H, W = proc0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*C["FOURCC"])
    out = cv2.VideoWriter(str(out_path), fourcc, out_fps, (W, H))
    if not out.isOpened():
        raise RuntimeError(f"No se pudo abrir VideoWriter: {out_path}")

    print(f"Procesando {in_path.name} -> {out_path.name} | {in_fps:.2f}fps → {out_fps:.2f}fps | salida: {W}x{H}")

    # Escribe primer frame
    out.write(proc0)

    # --- Pipeline paralelo ---
    # 1) Pre-cargar un buffer de frames (para no bloquear workers)
    # 2) Procesar en ThreadPool
    # 3) Escribir manteniendo orden (por índice)

    idx_next_write = 1
    pending = {}
    futures = deque()
    pool = ThreadPoolExecutor(max_workers=C["NUM_WORKERS"])

    def submit(idx, fr):
        fut = pool.submit(ctx.process, fr)
        futures.append((idx, fut))

    # Prefetch inicial
    prefetch = C["READER_QUEUE_SIZE"]
    idx_read = 1  # ya consumimos 0
    for _ in range(prefetch):
        ret, fr = cap.read()
        if not ret:
            break
        submit(idx_read, fr)
        idx_read += 1

    # Bucle: ir leyendo+procesando y escribiendo en orden
    while futures:
        # Saca completados sin bloquear demasiado
        idx, fut = futures.popleft()
        if fut.done():
            frame = fut.result()
            pending[idx] = frame
        else:
            # aún no termina: re-enfila al final
            futures.append((idx, fut))

        # Ir lanzando más trabajos si hay espacio
        while len(futures) < prefetch:
            ret, fr = cap.read()
            if not ret:
                break
            submit(idx_read, fr)
            idx_read += 1

        # Escribir en orden
        while idx_next_write in pending:
            out.write(pending.pop(idx_next_write))
            if total > 0 and idx_next_write % 100 == 0:
                print(f" {idx_next_write}/{total} frames ({100.0*idx_next_write/total:.1f}%)")
            elif total < 0 and idx_next_write % 200 == 0:
                print(f" {idx_next_write} frames...")
            idx_next_write += 1

        # Salir si ya no quedan tareas activas ni por leer
        if not futures and (idx_read > (total - 1) if total > 0 else True):
            break

    pool.shutdown(wait=True)
    out.release()
    cap.release()
    print("Listo. Usa este archivo como VIDEO_PATH en tu script principal.")


if __name__ == "__main__":
    main()
