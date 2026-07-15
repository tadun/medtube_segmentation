"""Evaluate RF-DETR on the MedTube 2 test split via Roboflow Inference API.

Computes Box mAP and Mask mAP by rasterising the polygon masks returned
by the inference SDK.

Usage:
    export ROBOFLOW_API_KEY="your_key_here"
    rs_env/bin/python notebooks/eval_rfdetr.py

Requires:
    pip install inference supervision
"""

import os
import json
import time
import contextlib
from pathlib import Path

os.environ["CORE_MODEL_GAZE_ENABLED"]  = "False"
os.environ["CORE_MODEL_SAM_ENABLED"]   = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"]  = "False"

import torch
if not torch.cuda.is_available():
    @contextlib.contextmanager
    def _noop_stream(stream):
        yield
    torch.cuda.stream = _noop_stream

import cv2
import numpy as np
import supervision as sv
from inference import get_model

MODEL_ID    = "medtube-2/1"
TEST_IMAGES = Path("/Users/tadun/Downloads/MedTube 2.yolov8/test/images")
TEST_LABELS = Path("/Users/tadun/Downloads/MedTube 2.yolov8/test/labels")
CONF        = 0.25
SAVE_DIR    = Path("runs/test_results_v2/RF-DETR-test")

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set ROBOFLOW_API_KEY environment variable before running.")

SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Loading RF-DETR model: {MODEL_ID}")
model = get_model(MODEL_ID, api_key=API_KEY)

image_paths = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
print(f"Evaluating on {len(image_paths)} test images...")


def poly_to_mask(pts_xy, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts_xy) >= 3:
        poly = np.array(pts_xy, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], (1,))
    return mask.astype(bool)


all_predictions, all_targets, inference_times = [], [], []

for img_path in image_paths:
    frame = cv2.imread(str(img_path))
    h, w  = frame.shape[:2]

    t0 = time.perf_counter()
    results = model.infer(frame, confidence=CONF)[0]
    inference_times.append((time.perf_counter() - t0) * 1000)

    pred_boxes, pred_confs, pred_labels, pred_masks = [], [], [], []
    for p in results.predictions:
        pred_boxes.append([p.x - p.width/2, p.y - p.height/2,
                           p.x + p.width/2, p.y + p.height/2])
        pred_confs.append(p.confidence)
        pred_labels.append(p.class_id)
        pts = [(pt.x, pt.y) for pt in (p.points or [])]
        pred_masks.append(poly_to_mask(pts, h, w))

    if pred_boxes:
        preds = sv.Detections(
            xyxy=np.array(pred_boxes, dtype=np.float32),
            confidence=np.array(pred_confs, dtype=np.float32),
            class_id=np.array(pred_labels, dtype=int),
            mask=np.stack(pred_masks),
        )
    else:
        preds = sv.Detections(
            xyxy=np.empty((0,4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
            mask=np.empty((0,h,w), dtype=bool),
        )
    all_predictions.append(preds)

    label_path = TEST_LABELS / (img_path.stem + ".txt")
    gt_boxes, gt_labels, gt_masks = [], [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls    = int(parts[0])
                coords = list(map(float, parts[1:]))
                xs = [coords[i]*w for i in range(0, len(coords), 2)]
                ys = [coords[i]*h for i in range(1, len(coords), 2)]
                gt_boxes.append([min(xs), min(ys), max(xs), max(ys)])
                gt_labels.append(cls)
                gt_masks.append(poly_to_mask([(int(x),int(y)) for x,y in zip(xs,ys)], h, w))

    if gt_boxes:
        gt = sv.Detections(
            xyxy=np.array(gt_boxes, dtype=np.float32),
            class_id=np.array(gt_labels, dtype=int),
            mask=np.stack(gt_masks),
        )
    else:
        gt = sv.Detections(
            xyxy=np.empty((0,4), dtype=np.float32),
            class_id=np.empty(0, dtype=int),
            mask=np.empty((0,h,w), dtype=bool),
        )
    all_targets.append(gt)

box_ap = sv.MeanAveragePrecision.from_detections(all_predictions, all_targets)

try:
    mask_ap       = sv.MeanAveragePrecision.from_detections(all_predictions, all_targets, use_mask_iou=True)
    mask_map50    = mask_ap.map50
    mask_map50_95 = mask_ap.map50_95
except TypeError:
    mask_map50 = mask_map50_95 = None

avg_ms = float(np.mean(inference_times))

print("\n--- RF-DETR Test-Split Results ---")
print(f"Box  mAP50    : {box_ap.map50:.4f}")
print(f"Box  mAP50-95 : {box_ap.map50_95:.4f}")
if mask_map50 is not None:
    print(f"Mask mAP50    : {mask_map50:.4f}")
    print(f"Mask mAP50-95 : {mask_map50_95:.4f}")
else:
    print("Mask mAP      : supervision version does not support use_mask_iou")
print(f"Inference     : {avg_ms:.1f} ms/image")

result = {
    "model": MODEL_ID, "dataset": "MedTube 2 test split", "images": len(image_paths),
    "box_map50": round(box_ap.map50, 4), "box_map50_95": round(box_ap.map50_95, 4),
    "mask_map50": round(mask_map50, 4) if mask_map50 is not None else None,
    "mask_map50_95": round(mask_map50_95, 4) if mask_map50_95 is not None else None,
    "inference_ms": round(avg_ms, 1),
}
(SAVE_DIR / "results.json").write_text(json.dumps(result, indent=2))
print(f"Results saved to: {SAVE_DIR}/results.json")
