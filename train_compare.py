"""
YOLO segmentation model comparison — MedTube dataset
======================================================
Trains YOLOv8-seg, YOLOv9-seg and YOLOv11-seg on the same dataset with the
same augmentation settings, then prints a side-by-side mAP summary.

Usage
-----
1. Annotate your images with instance segmentation masks (Roboflow, CVAT, or
    Label Studio) and export in YOLO segmentation format.
2. Place or symlink your dataset under data/  with this structure:
         data/
            dataset.yaml
            train/
              images/
              labels/
            val/
              images/
              labels/
3. Run:
         rs_env/bin/python train_compare.py
    (no sudo needed — training does not use the RealSense camera)

Install dependencies first:
    rs_env/bin/pip install ultralytics torch torchvision
"""

import json
import os
from datetime import datetime
from pathlib import Path

_ULTRALYTICS_CONFIG_DIR = Path(__file__).resolve().parent / ".ultralytics"
_ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_YAML   = "data/dataset.yaml"   # path to your dataset YAML
EPOCHS      = 100
IMG_SIZE    = 640
BATCH       = 8                     # lower if you get OOM errors
PATIENCE    = 20                    # early-stopping patience (epochs)
WORKERS     = 4
DEVICE      = ""                    # "" = auto (MPS on Apple Silicon, CUDA if available, else CPU)

# Models to compare — keys are display names, values are Ultralytics model IDs
MODELS = {
    "YOLOv8-seg": "yolov8m-seg.pt",
    "YOLOv9-seg": "yolov9c-seg.pt",
    "YOLOv11-seg": "yolo11m-seg.pt",
}

# ── Augmentation ──────────────────────────────────────────────────────────────
# Tuned for a controlled studio setup:
#   - black matte background, ring light → aggressive colour/brightness jitter OK
#   - tubes are non-overlapping and horizontal → strong rotation/flip safe
#   - small dataset → mosaic and mixup help generalisation

AUG = dict(
    hsv_h      = 0.015,   # hue jitter  (subtle — tubes have distinct cap colours)
    hsv_s      = 0.7,     # saturation jitter
    hsv_v      = 0.4,     # brightness/value jitter
    degrees    = 180.0,   # full rotation (tubes are rotationally symmetric)
    translate  = 0.1,     # random translation fraction
    scale      = 0.5,     # random scale ±50%
    shear      = 5.0,     # mild shear
    perspective= 0.0005,  # subtle perspective warp
    flipud     = 0.5,     # vertical flip
    fliplr     = 0.5,     # horizontal flip
    mosaic     = 1.0,     # mosaic augmentation (combines 4 images)
    mixup      = 0.1,     # mixup blending
    copy_paste = 0.1,     # copy-paste augmentation (good for instance seg)
    erasing    = 0.4,     # random erasing (simulates partial occlusion)
)

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(name: str, weights: str, run_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training {name}  ({weights})")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH,
        patience   = PATIENCE,
        workers    = WORKERS,
        device     = DEVICE,
        project    = str(run_dir),
        name       = name,
        exist_ok   = True,
        **AUG,
    )

    # Extract validation metrics — fall back to an explicit val() run if
    # train() returned None (can happen on early stop or resume).
    if results is not None and hasattr(results, "results_dict"):
        metrics = results.results_dict
    else:
        metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, device=DEVICE).results_dict
    return {
        "model":          name,
        "weights":        weights,
        "mAP50":          round(metrics.get("metrics/mAP50(M)",    0.0), 4),
        "mAP50-95":       round(metrics.get("metrics/mAP50-95(M)", 0.0), 4),
        "precision":      round(metrics.get("metrics/precision(M)", 0.0), 4),
        "recall":         round(metrics.get("metrics/recall(M)",   0.0), 4),
        "best_weights":   str(run_dir / name / "weights" / "best.pt"),
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    col_w = 14
    headers = ["Model", "mAP50", "mAP50-95", "Precision", "Recall"]
    header_line = "  ".join(h.ljust(col_w) for h in headers)
    print(f"\n{'='*len(header_line)}")
    print("  MODEL COMPARISON RESULTS")
    print(f"{'='*len(header_line)}")
    print(header_line)
    print("-" * len(header_line))
    for r in sorted(results, key=lambda x: x["mAP50-95"], reverse=True):
        row = [r["model"], str(r["mAP50"]), str(r["mAP50-95"]),
               str(r["precision"]), str(r["recall"])]
        print("  ".join(v.ljust(col_w) for v in row))
    print(f"{'='*len(header_line)}\n")

    best = max(results, key=lambda x: x["mAP50-95"])
    print(f"  Best model: {best['model']}  (mAP50-95 = {best['mAP50-95']})")
    print(f"  Best weights: {best['best_weights']}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"[error] Dataset YAML not found: {DATA_YAML}")
        print("  Annotate your images and export in YOLO segmentation format,")
        print("  then place dataset.yaml under data/")
        return

    run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    failed = []

    for name, weights in MODELS.items():
        try:
            result = train_model(name, weights, run_dir)
            all_results.append(result)
        except Exception as e:
            print(f"[error] {name} failed: {e}")
            failed.append(name)

    if all_results:
        print_summary(all_results)

        summary_path = run_dir / "comparison.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Full results saved to {summary_path}")

    if failed:
        print(f"\n[warn] These models failed to train: {', '.join(failed)}")


if __name__ == "__main__":
    main()
