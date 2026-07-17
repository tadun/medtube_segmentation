"""Multi-model stream — runs 4 YOLO models in parallel on the same camera feed.

Each model gets its own OpenCV window showing the segmentation overlay,
allowing side-by-side visual comparison at the same instant.

Usage:
    sudo rs_env/bin/python src/stream_multimodel.py

Note: Runs at the FPS of the slowest model (~3 FPS due to YOLOv8m).
"""

import os
import sys
import time
import contextlib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_ULTRALYTICS_CONFIG_DIR = _PROJECT_ROOT / ".ultralytics"
_ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR = _PROJECT_ROOT / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# ── Models to compare ─────────────────────────────────────────────────────────
MODELS = {
    "YOLO26n": _PROJECT_ROOT / "weights" / "yolo26n.pt",
    "YOLO11n": _PROJECT_ROOT / "weights" / "yolo11n.pt",
    "YOLOv8m": _PROJECT_ROOT / "runs" / "segment" / "runs" / "2026-07-12_22-48-54" / "YOLOv8-seg" / "weights" / "best.pt",
    "YOLOv9c": _PROJECT_ROOT / "YOLOv9c-seg" / "weights" / "best.pt",
}

# ── Class colours (BGR) ───────────────────────────────────────────────────────
CLASS_COLORS = {
    "Other":     (  0, 220, 220),
    "Push-on":   (220,   0,   0),
    "Screwcap":  (  0, 200,   0),
    "Universal": (  0,   0, 220),
}
MASK_ALPHA = 0.55
CONF = 0.35
IMGSZ = 640

# ── Camera config ─────────────────────────────────────────────────────────────
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30


def draw_overlay(image: np.ndarray, result, model_name: str) -> np.ndarray:
    """Draw masks + boxes on the image with model name label."""
    overlay = image.copy()
    if result.masks is not None and len(result.masks) > 0:
        for seg_xy, box in zip(result.masks.xy, result.boxes):
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            color = CLASS_COLORS.get(cls_name, (180, 60, 255))

            # Draw mask
            pts = seg_xy.astype(np.int32).reshape(-1, 1, 2)
            mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            coloured = np.full_like(overlay, color, dtype=np.uint8)
            overlay = np.where(mask[:, :, None] > 0,
                               cv2.addWeighted(overlay, 1 - MASK_ALPHA, coloured, MASK_ALPHA, 0),
                               overlay)

            # Draw box + label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    # Model name + FPS label at top-left
    cv2.putText(overlay, model_name, (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def main():
    # Load all models
    print("\033[36m[info]\033[0m Loading models...")
    models = {}
    for name, path in MODELS.items():
        if path.exists():
            models[name] = YOLO(str(path))
            print(f"  \033[32m✓\033[0m {name}")
        else:
            print(f"  \033[33m✗\033[0m {name} — not found at {path}")

    if not models:
        print("\033[31m[error]\033[0m No models found.")
        return

    # Start camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    profile = pipeline.start(config)

    print("\033[36m[info]\033[0m Waiting 4s for auto-exposure...")
    time.sleep(4.0)

    # Create windows — arrange in a 2×2 grid
    win_names = list(models.keys())
    positions = [(0, 0), (760, 0), (0, 420), (760, 420)]
    for i, name in enumerate(win_names):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if i < len(positions):
            cv2.moveWindow(name, *positions[i])

    print("\033[36m[info]\033[0m Streaming — press Q to quit")
    frame_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.flip(frame, -1)  # camera mounted upside-down

            # Run each model and show in its window
            for name, model in models.items():
                result = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
                overlay = draw_overlay(frame, result, name)
                # Scale for Retina
                display = cv2.resize(overlay, (1512, 850))
                cv2.imshow(name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_dir = _PROJECT_ROOT / "runs" / "captures" / "multimodel"
                save_dir.mkdir(parents=True, exist_ok=True)
                for name, model in models.items():
                    result = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
                    overlay = draw_overlay(frame, result, name)
                    cv2.imwrite(str(save_dir / f"{name}_{ts}.png"), overlay)
                print(f"\033[36m[info]\033[0m Multi-model snapshot saved → {save_dir}")

            frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n\033[1m{'═' * 40}\033[0m")
        print(f"  Multi-model stream ended ({frame_count} frames)")
        print(f"\033[1m{'═' * 40}\033[0m")


if __name__ == "__main__":
    main()
