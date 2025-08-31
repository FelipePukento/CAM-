import os
import cv2
import time
import csv
import torch
import numpy as np
from collections import deque, Counter
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# --------- OPTIMIZACIÓN/RUNTIME ---------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =========================
# CONFIG (editar aquí)
# =========================
CONFIG = {
    # Fuente de video
    "USE_VIDEO_FILE": True,
    "VIDEO_PATH": r"video/Clase-R_preproc.mp4",
    "CAM_INDEX": 0,
    "MIRROR": False,

    # Objetivo de tiempo real
    "TARGET_FPS": 30.0,
    "REALTIME_FROM_FILE": True,

    # Redimensionamiento dinámico (lado corto objetivo)
    "DOWNSCALE_STEPS": [1080,720],
    "START_STEP_INDEX": 2,
    "UPSCALE_HEADROOM": 1.15,
    "DOWNSCALE_PRESSURE": 0.90,
    "IMG_SIZE_INFER": 896,  # usado en algunas rutas; el DET usa valores mayores abajo

    # Ventana
    "WINDOW_TITLE": "Atención grupal (YOLOv11 2-etapas)",
    "ALLOW_RESIZE_WINDOW": True,
    "SHOW_ROI": True,

    # Modelos
    "MODEL_POSE": "models/yolo11x-pose.pt",
    "MODEL_DET":  "models/yolo11x.pt",
    "CONF_POSE":  0.35,
    "CONF_DET":   0.20,
    "USE_PHONE":  True,

    # ROI (x,y,w,h) relativo (0-1)
    "ROI": [0.28, 0.05, 0.44, 0.65],

    # Pesos/umbrales base (toma lejana)
    "W_FRONTAL": 15,
    "W_HEAD":    20,
    "W_TORSO":   40,
    "W_HANDS":   25,
    "THR_FRONTAL": 0.25,
    "THR_HEAD":    30.0,
    "THR_TORSO":   15.0,
    "THR_HANDS":   0.50,

    # Extras (temporales)
    "THR_YAW":   0.45,
    "THR_SPEED": 0.09,
    "WIN_SECS":  3.0,
    "FPS_HINT":  60.0,

    # Penalizaciones / bonos
    "PEN_YAW":   8.0,
    "PEN_SPEED": 8.0,
    "BONUS_ROI": 6.0,
    "PEN_PHONE": 12.0,

    # Agregación (no mostramos por persona)
    "EMA":         0.85,
    "HI":          75.0,
    "LO":          50.0,
    "HOLD_FRAMES": 12,

    # CSV y resumen
    "CSV_PATH": "",
    "SUMMARY_SECS": 60.0,
}

# --------- RUTAS SEGURAS ---------
ROOT = Path(__file__).resolve().parent
if CONFIG["USE_VIDEO_FILE"]:
    CONFIG["VIDEO_PATH"] = str((ROOT / CONFIG["VIDEO_PATH"]).resolve())

# =========================
# Utilidades
# =========================
def angle_to_vertical(p1, p2):
    v = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.nan
    cosang = (v @ np.array([0.0, 1.0])) / (n + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))  # 0° = vertical

def dist(p, q):
    return float(np.linalg.norm(np.array(p, dtype=float) - np.array(q, dtype=float)))

def mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

def draw_label(img, text, org, scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

def clamp01(x):
    return max(0.0, min(1.0, x))

# =========================
# Features desde keypoints
# =========================
def compute_features(person_xy):
    # COCO-17
    nose = person_xy[0]
    eyeL, eyeR = person_xy[1], person_xy[2]
    shL, shR = person_xy[5], person_xy[6]
    wrL, wrR = person_xy[9], person_xy[10]
    hipL, hipR = person_xy[11], person_xy[12]

    sh_mid = mid(shL, shR)
    hip_mid = mid(hipL, hipR)
    shoulder_w = max(dist(shL, shR), 1.0)

    eye_span = 0.0 if (np.all(eyeL == 0) or np.all(eyeR == 0)) else dist(eyeL, eyeR)
    frontal_ratio = eye_span / shoulder_w

    head_angle = angle_to_vertical(sh_mid, nose)
    torso_angle = angle_to_vertical(hip_mid, sh_mid)
    hand_face_min = (min(dist(wrL, nose), dist(wrR, nose)) / shoulder_w if not np.all(nose == 0) else 1.0)
    yaw_proxy = (abs(nose[0] - sh_mid[0]) / shoulder_w if not np.all(nose == 0) else 0.0)

    return {
        "frontal_ratio": frontal_ratio,
        "head_angle": head_angle,
        "torso_angle": torso_angle,
        "hand_face_min": hand_face_min,
        "yaw_proxy": yaw_proxy,
        "shoulder_w": shoulder_w,
        "sh_mid": sh_mid,
        "hip_mid": hip_mid,
        "nose": nose,
        "wrL": wrL,
        "wrR": wrR,
    }

def score_base(feat, P):
    s = 0.0
    s += P["W_FRONTAL"] if feat["frontal_ratio"] >= P["THR_FRONTAL"] else P["W_FRONTAL"] * 0.2
    s += P["W_HEAD"] if (not np.isnan(feat["head_angle"]) and feat["head_angle"] <= P["THR_HEAD"]) else P["W_HEAD"] * 0.3
    s += P["W_TORSO"] if (not np.isnan(feat["torso_angle"]) and feat["torso_angle"] <= P["THR_TORSO"]) else P["W_TORSO"] * 0.4
    s += P["W_HANDS"] if feat["hand_face_min"] >= P["THR_HANDS"] else P["W_HANDS"] * 0.35
    return float(np.clip(s, 0, 100))

# =========================
# Carga modelos
# =========================
USE_CUDA = torch.cuda.is_available()
USE_HALF = bool(USE_CUDA)
print("CUDA disponible:", USE_CUDA)
if USE_CUDA:
    try:
        print("GPU usada:", torch.cuda.get_device_name(0))
    except Exception:
        pass

pose_model = YOLO(CONFIG["MODEL_POSE"]).to("cuda" if USE_CUDA else "cpu")
det_model  = YOLO(CONFIG["MODEL_DET"]).to("cuda" if USE_CUDA else "cpu")

# =========================
# Wrappers / 2-etapas
# =========================
def person_boxes(frame_bgr, imgsz_det=1280, conf_det=0.20, max_det=600, iou=0.5):
    """Detección solo de 'person' (clase 0) en alta resolución."""
    r = det_model(
        frame_bgr,
        device=0 if USE_CUDA else "cpu",
        imgsz=imgsz_det,
        conf=conf_det,
        iou=iou,
        classes=[0],        # SOLO persona
        max_det=max_det,
        half=USE_HALF,
        verbose=False,
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)

def person_boxes_tiled(frame_bgr, tiles=2, **det_kwargs):
    """Fallback: tiling 2x2 para simular zoom en alumnos muy lejanos."""
    H, W = frame_bgr.shape[:2]
    th, tw = H // tiles, W // tiles
    boxes_all = []
    for r in range(tiles):
        for c in range(tiles):
            y0, y1 = r*th, (r+1)*th if r<tiles-1 else H
            x0, x1 = c*tw, (c+1)*tw if c<tiles-1 else W
            crop = frame_bgr[y0:y1, x0:x1]
            b = person_boxes(crop, **det_kwargs)
            if len(b) == 0:
                continue
            b[:, [0,2]] += x0
            b[:, [1,3]] += y0
            boxes_all.append(b)
    if not boxes_all:
        return np.zeros((0,4), dtype=np.float32)
    boxes = np.vstack(boxes_all)
    # deduplicación simple por IoU
    keep = []
    used = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if used[i]: continue
        keep.append(i)
        x1,y1,x2,y2 = boxes[i]
        area_i = (x2-x1)*(y2-y1)
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            X1 = max(x1, boxes[j,0]); Y1 = max(y1, boxes[j,1])
            X2 = min(x2, boxes[j,2]); Y2 = min(y2, boxes[j,3])
            iw, ih = max(0, X2-X1), max(0, Y2-Y1)
            inter = iw*ih
            area_j = (boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])
            iou = inter / max(area_i + area_j - inter, 1e-6)
            if iou > 0.6:
                used[j] = True
    return boxes[keep]

def pose_on_crops(frame_bgr, boxes_xyxy, expand=0.10, pose_imgsz=384, conf_pose=None):
    """Corre POSE en cada recorte y remapea a coords globales."""
    H, W = frame_bgr.shape[:2]
    kps_out = []
    conf_pose = conf_pose if conf_pose is not None else CONFIG["CONF_POSE"]
    for (x1, y1, x2, y2) in boxes_xyxy:
        w = x2 - x1; h = y2 - y1
        cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5
        x1e = max(0, int(cx - (1+expand)*w*0.5))
        x2e = min(W, int(cx + (1+expand)*w*0.5))
        y1e = max(0, int(cy - (1+expand)*h*0.5))
        y2e = min(H, int(cy + (1+expand)*h*0.5))
        crop = frame_bgr[y1e:y2e, x1e:x2e]
        if crop.size == 0:
            kps_out.append(None); continue
        r = pose_model(
            crop,
            device=0 if USE_CUDA else "cpu",
            imgsz=pose_imgsz,
            conf=conf_pose,
            iou=0.6,
            max_det=1,
            half=USE_HALF,
            verbose=False,
        )[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            kps_out.append(None); continue
        kp = r.keypoints.xy[0].detach().cpu().numpy()  # (17,2) local
        kp[:, 0] += x1e; kp[:, 1] += y1e
        kps_out.append(kp)
    return kps_out

def infer_det_for_phone(img_bgr):
    """Detección de 'person' + 'cell phone' para alertas de teléfono."""
    r = det_model(
        img_bgr,
        device=0 if USE_CUDA else "cpu",
        imgsz=max(1024, CONFIG["IMG_SIZE_INFER"]),
        conf=CONFIG["CONF_DET"],
        iou=0.5,
        max_det=800,
        classes=[0, 67],        # persona y teléfono
        half=USE_HALF,
        verbose=False,
    )[0]
    return r

# =========================
# Adaptador de reproducción en tiempo real
# =========================
class RealTimeFileSync:
    def __init__(self, cap, target_fps):
        self.cap = cap
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        if self.video_fps <= 0:
            self.video_fps = target_fps
        self.start_wall = None
        self.frame_idx = 0
    def start(self):
        self.start_wall = time.time()
        self.frame_idx = 0
    def gate(self):
        if self.start_wall is None:
            self.start()
        now = time.time()
        should_be = self.frame_idx / self.video_fps
        wall_elapsed = now - self.start_wall
        delay = should_be - wall_elapsed
        if delay > 0.0:
            time.sleep(min(delay, 0.010))
            return True
        else:
            behind_frames = int((-delay) * self.video_fps)
            if behind_frames >= 2:
                for _ in range(max(0, behind_frames - 1)):
                    ok, _ = self.cap.read()
                    if not ok:
                        return False
                    self.frame_idx += 1
            return True
    def step(self):
        self.frame_idx += 1

# =========================
# Main
# =========================
def main():
    P = CONFIG

    # Fuente de video
    if P["USE_VIDEO_FILE"]:
        cap = cv2.VideoCapture(P["VIDEO_PATH"], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        sync = RealTimeFileSync(cap, P["TARGET_FPS"]) if P["REALTIME_FROM_FILE"] else None
    else:
        cap = cv2.VideoCapture(P["CAM_INDEX"], cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, int(P["TARGET_FPS"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        sync = None

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la fuente de video.")

    if P["ALLOW_RESIZE_WINDOW"]:
        cv2.namedWindow(P["WINDOW_TITLE"], cv2.WINDOW_NORMAL)

    # CSV global (agregados por frame)
    writer = None
    fcsv = None
    if P["CSV_PATH"]:
        fcsv = open(P["CSV_PATH"], "w", newline="", encoding="utf-8")
        writer = csv.writer(fcsv)
        writer.writerow([
            "timestamp","num_personas","score_promedio",
            "%ALTA","%MEDIA","%BAJA","phone_near_group","scale_shorter",
        ])

    fps_avg, t0 = 0.0, time.time()
    H = W = None

    # Downscale dinámico
    steps = P["DOWNSCALE_STEPS"]
    step_idx = max(0, min(P["START_STEP_INDEX"], len(steps) - 1))
    current_short = steps[step_idx]

    # Agregación temporal
    summary_start = time.time()
    reasons_hist = Counter()
    level_hist = Counter()

    while True:
        if P["REALTIME_FROM_FILE"] and sync and not sync.gate():
            break

        ok, frame = cap.read()
        if not ok:
            break
        if P["MIRROR"]:
            frame = cv2.flip(frame, 1)
        if H is None or W is None:
            H, W = frame.shape[:2]

        # Redimensionar a "lado corto" actual (mantener aspecto)
        h, w = frame.shape[:2]
        short_side = min(h, w)
        scale = current_short / short_side
        if scale != 1.0:
            new_w = int(w * scale); new_h = int(h * scale)
            frame_inf = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_inf = frame

        # ========= 2-ETAPAS: DET (person) -> POSE por recorte =========
        # 1) DET sensible (alta resolución); fallback tiled si es necesario
        boxes = person_boxes(
            frame_inf,
            imgsz_det=max(1280, CONFIG["IMG_SIZE_INFER"]),
            conf_det=CONFIG["CONF_DET"],
            max_det=600,
            iou=0.5
        )
        if len(boxes) == 0:
            boxes = person_boxes_tiled(
                frame_inf, tiles=2,
                imgsz_det=1024, conf_det=max(0.18, CONFIG["CONF_DET"]-0.02),
                max_det=800, iou=0.5
            )

        # 2) POSE por recorte (rápido, por persona)
        kps_list = pose_on_crops(
            frame_inf, boxes,
            expand=0.10, pose_imgsz=384, conf_pose=CONFIG["CONF_POSE"]
        )

        # 3) Dibujo simple (cajas + puntos de keypoints)
        annotated = frame_inf.copy()
        for (b, kp) in zip(boxes.astype(int), kps_list):
            x1, y1, x2, y2 = b
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
            if kp is not None:
                for (x, y) in kp.astype(int):
                    if x > 0 and y > 0:
                        cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

        # Calcular ROI en coordenadas actuales
        Hc, Wc = annotated.shape[:2]
        rx, ry, rw, rh = P["ROI"]
        roi_px = (int(Wc * rx), int(Hc * ry), int(Wc * rw), int(Hc * rh))
        if P["SHOW_ROI"]:
            x, y, wroi, hroi = roi_px
            cv2.rectangle(annotated, (x, y), (x + wroi, y + hroi), (255, 255, 255), 1)

        # Teléfono (global): reutiliza detector con clases [0,67]
        phone_near_group = False
        phone_centers = []
        if P["USE_PHONE"]:
            res_det = infer_det_for_phone(frame_inf)
            if res_det and res_det.boxes is not None and len(res_det.boxes) > 0:
                boxes_all = res_det.boxes.xyxy.cpu().numpy()
                clses = res_det.boxes.cls.cpu().numpy().astype(int)
                for b, c in zip(boxes_all, clses):
                    if c == 67:  # cell phone
                        x1, y1, x2, y2 = b
                        phone_centers.append(np.array([(x1 + x2) / 2, (y1 + y2) / 2]))

        # Procesar todas las personas y agregar
        num_people = 0
        per_scores = []
        per_levels = []

        if kps_list:
            valid_kps = []
            for kp in kps_list:
                if kp is None:
                    continue
                if np.all(kp[5] == 0) or np.all(kp[6] == 0):  # requiere hombros
                    continue
                valid_kps.append(kp)

            num_people = len(valid_kps)

            for person in valid_kps:
                feat = compute_features(person)
                # ROI flag
                x, y, wroi, hroi = roi_px
                in_roi = (x <= feat["nose"][0] <= x + wroi) and (y <= feat["nose"][1] <= y + hroi)
                yaw_over = 1.0 if feat["yaw_proxy"] > CONFIG["THR_YAW"] else 0.0

                # Score base
                s0 = score_base(feat, CONFIG)
                s = s0
                reasons = []
                if yaw_over > 0.5:
                    s -= CONFIG["PEN_YAW"]
                    reasons.append("yaw")
                if in_roi:
                    s += CONFIG["BONUS_ROI"]
                    reasons.append("roi")

                # Phone cerca (por persona)
                if P["USE_PHONE"] and phone_centers:
                    wrL = np.array(feat["wrL"]); wrR = np.array(feat["wrR"]); nose = np.array(feat["nose"])
                    for c in phone_centers:
                        if (
                            np.linalg.norm(c - wrL) < 0.6 * feat["shoulder_w"]
                            or np.linalg.norm(c - wrR) < 0.6 * feat["shoulder_w"]
                            or np.linalg.norm(c - nose) < 0.7 * feat["shoulder_w"]
                        ):
                            s = max(0.0, s - CONFIG["PEN_PHONE"])
                            phone_near_group = True
                            reasons.append("phone")
                            break

                per_scores.append(s)
                lvl = ("ALTA" if s >= CONFIG["HI"] else "MEDIA" if s >= CONFIG["LO"] else "BAJA")
                per_levels.append(lvl)
                for r in reasons:
                    reasons_hist[r] += 1

        # Agregados de grupo
        if per_scores:
            score_mean = float(np.mean(per_scores))
            cnt = Counter(per_levels)
            n = max(1, len(per_scores))
            p_alta = 100.0 * cnt["ALTA"] / n
            p_media = 100.0 * cnt["MEDIA"] / n
            p_baja = 100.0 * cnt["BAJA"] / n

            # Panel global
            x0, y0 = 10, 26
            draw_label(annotated, f"Personas: {n}", (x0, y0))
            draw_label(annotated, f"Score Prom.: {score_mean:5.1f}/100", (x0, y0 + 24))
            draw_label(annotated, f"ALTA:{p_alta:4.1f}%  MEDIA:{p_media:4.1f}%  BAJA:{p_baja:4.1f}%", (x0, y0 + 48))
            if phone_near_group:
                draw_label(annotated, f"Alerta: uso de teléfono detectado", (x0, y0 + 72))

            # CSV
            if writer:
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    n, f"{score_mean:.2f}", f"{p_alta:.2f}", f"{p_media:.2f}", f"{p_baja:.2f}",
                    int(phone_near_group), int(current_short),
                ])

            # Resumen cada SUMMARY_SECS
            level_hist.update(cnt)
            if time.time() - summary_start >= P["SUMMARY_SECS"] and sum(level_hist.values()) > 0:
                total = sum(level_hist.values())
                s_alta = 100.0 * level_hist["ALTA"] / total
                s_media = 100.0 * level_hist["MEDIA"] / total
                s_baja = 100.0 * level_hist["BAJA"] / total
                top_reasons = ", ".join([f"{k}:{v}" for k, v in reasons_hist.most_common(3)])
                summary_text = (
                    f"{P['SUMMARY_SECS']:.0f}s -> ALTA:{s_alta:.1f}%  MEDIA:{s_media:.1f}%  BAJA:{s_baja:.1f}% | Motivos: {top_reasons}"
                )
                print(summary_text)
                draw_label(annotated, summary_text, (10, annotated.shape[0] - 10), 0.55)
                summary_start = time.time()
                level_hist.clear()
                reasons_hist.clear()
        else:
            draw_label(annotated, "Sin personas válidas", (10, 26))

        # FPS y ajuste dinámico de escala
        fps_inst = 1.0 / max(time.time() - t0, 1e-6)
        fps_avg = 0.9 * fps_avg + 0.1 * fps_inst if 'fps_avg' in locals() and fps_avg > 0 else fps_inst
        t0 = time.time()
        draw_label(annotated, f"FPS: {fps_avg:4.1f}  • short={current_short}px", (10, annotated.shape[0] - 10))

        # Adaptar escala
        if fps_avg < P["TARGET_FPS"] * P["DOWNSCALE_PRESSURE"] and step_idx < len(steps) - 1:
            step_idx += 1
            current_short = steps[step_idx]
        elif fps_avg > P["TARGET_FPS"] * P["UPSCALE_HEADROOM"] and step_idx > 0:
            step_idx -= 1
            current_short = steps[step_idx]

        # Mostrar
        cv2.imshow(P["WINDOW_TITLE"], annotated)
        if P["REALTIME_FROM_FILE"] and sync:
            sync.step()
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    if fcsv:
        fcsv.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
