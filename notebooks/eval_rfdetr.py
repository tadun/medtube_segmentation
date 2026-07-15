"""Evaluate RF-DETR on the MedTube 2 test split via Roboflow Inference API.

Usage:
    export ROBOFLOW_API_KEY="your_key_here"
    rs_env/bin/python notebooks/eval_rfdetr.py

Requires:
    pip install inference supervision
"""

import os
import json
from pathlib import Path

# Force CPU-only execution — prevents torch.cuda.stream crash on Apple Silicon MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"

import cv2
import numpy as np
import supervision as sv
from inference import get_model

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "medtube-2/1"
TEST_IMAGES = Path("/Users/tadun/Downloads/MedTube 2.yolov8/test/images")
TEST_LABELS = Path("/Users/tadun/Downloads/MedTube 2.yolov8/test/labels")
CLASSES     = ["Other", "Push-on", "Screwcap", "Universal"]
CONF        = 0.25
SAVE_DIR    = Path("runs/test_results_v2/RF-DETR-test")
# ──────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set ROBOFLOW_API_KEY environment variable before running.")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading RF-DETR model: {MODEL_ID}")
model = get_model(MODEL_ID, api_key=API_KEY)

image_paths = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
print(f"Evaluating on {len(image_paths)} test images...")

all_predictions = []
all_targets     = []
inference_times = []

for img_path in image_paths:
    frame = cv2.imread(str(img_path))
    h, w  = frame.shape[:2]

    import time
    t0 = time.perf_counter()
    results = model.infer(frame, confidence=CONF)[0]
    inference_times.append((time.perf_counter() - t0) * 1000)

    # Convert predictions to supervision Detections
    pred_boxes  = np.array([[p.x - p.width/2, p.y - p.height/2,
                              p.x + p.width/2, p.y + p.height/2]
                             for p in results.predictions], dtype=np.float32) if results.predictions else np.empty((0, 4))
    pred_confs  = np.array([p.confidence for p in results.predictions], dtype=np.float32) if results.predictions else np.empty(0)
    pred_labels = np.array([p.class_id   for p in results.predictions], dtype=int) if results.predictions else np.empty(0, dtype=int)

    preds = sv.Detections(xyxy=pred_boxes, confidence=pred_confs, class_id=pred_labels)
    all_predictions.append(preds)

    # Load ground truth from YOLO label file
    label_path = TEST_LABELS / (img_path.stem + ".txt")
    gt_boxes, gt_labels = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(cls)
    gt = sv.Detections(
        xyxy=np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4)),
        class_id=np.array(gt_labels, dtype=int) if gt_labels else np.empty(0, dtype=int),
    )
    all_targets.append(gt)

# Compute mAP using supervision
mean_ap = sv.MeanAveragePrecision.from_detections(
    predictions=all_predictions,
    targets=all_targets,
)

avg_ms = np.mean(inference_times)

print("\n--- RF-DETR Test-Split Results ---")
print(f"Images evaluated : {len(image_paths)}")
print(f"Box mAP50        : {mean_ap.map50:.4f}")
print(f"Box mAP50-95     : {mean_ap.map50_95:.4f}")
print(f"Avg inference    : {avg_ms:.1f} ms/image")

# Save results
result = {
    "model": MODEL_ID,
    "dataset": "MedTube 2 test split",
    "images": len(image_paths),
    "box_map50":    round(mean_ap.map50, 4),
    "box_map50_95": round(mean_ap.map50_95, 4),
    "inference_ms": round(avg_ms, 1),
}
out = SAVE_DIR / "results.json"
out.write_text(json.dumps(result, indent=2))
print(f"\nResults saved to: {out}")
