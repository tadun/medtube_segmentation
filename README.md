# MedTube Segmentation

Dataset collection and instance segmentation pipeline for medical tubes using an Intel RealSense D415 depth camera.

---

## Hardware

| Component          | Details                  |
| ------------------ | ------------------------ |
| Camera             | Intel RealSense D415     |
| Connection         | USB 3.2                  |
| Capture resolution | 1280 × 720 @ 30 fps      |

> **macOS note:** `pyrealsense2` requires elevated privileges to claim the USB interface.
> Always prefix RealSense scripts with `sudo rs_env/bin/python`.

---

## Setup

```bash
# Create and activate the virtual environment
python3.12 -m venv rs_env
source rs_env/bin/activate

# Install dependencies
pip install pyrealsense2 opencv-python numpy ultralytics roboflow
```

---

## Scripts

### `collect_dataset.py` — Data collection

Streams aligned RGB and depth from the D415 and saves paired frames to disk.

```bash
sudo rs_env/bin/python collect_dataset.py
```

Prompts for a session label on startup (e.g. `single`, `pairs`, `mixed`).

| Key     | Action                              |
| ------- | ----------------------------------- |
| `Space` | Start countdown then record / pause |
| `S`     | Save a single frame immediately     |
| `Q`     | Quit                                |

#### Folder structure

```text
dataset/
  YYYY-MM-DD_HH-MM-SS_LABEL/
    rgb/          Colour PNGs  (1280 x 720, BGR)
    depth_raw/    16-bit PNGs  (Z16, millimetres)
    metadata.csv  frame index + Unix timestamp per saved frame
```

The live preview shows a normalised grayscale depth image (bright = close, dark = far, black = no IR return).
Raw 16-bit depth values are preserved in saved files.

---

### `realsense_stream.py` — Live preview

Minimal streaming script for verifying the camera feed.

```bash
sudo rs_env/bin/python realsense_stream.py
```

---

### `train_compare.py` — Model comparison

Trains YOLOv8-seg, YOLOv9-seg and YOLOv11-seg on the same dataset with identical augmentation settings and prints a side-by-side mAP summary.

```bash
rs_env/bin/python train_compare.py
```

Expects the dataset in YOLO segmentation format under `data/`:

```text
data/
  dataset.yaml
  train/
    images/
    labels/
  val/
    images/
    labels/
```

Export from Roboflow in **YOLOv8 format** — the download unpacks directly into this structure.

Built-in augmentation presets:

- `strong` (default): aggressive transforms for small datasets
- `mild`: conservative transforms to reduce synthetic shift
- `none`: disable augmentation for baseline comparison

---

## QA Utilities

Project viewers and inspection helpers live under `tools/` to keep the root focused on the main collection, annotation, and training pipeline.

```bash
# preview auto-filled YOLO labels
rs_env/bin/python tools/preview_autofilled_labels.py

# inspect COCO masks with class overlays
rs_env/bin/python tools/view_coco_masks.py --dataset data_coco --random
```

---

## Kaggle Training (Free GPU)

For long runs on free Kaggle sessions, use the single-model trainer with checkpoint resume:

```bash
python kaggle_train.py \
  --data /kaggle/input/<your-dataset>/data.yaml \
  --model yolo11m-seg.pt \
  --name medtube-yolo11 \
  --aug-preset mild
```

Resume after a Kaggle disconnect or session timeout:

```bash
python kaggle_train.py \
  --data /kaggle/input/<your-dataset>/data.yaml \
  --name medtube-yolo11 \
  --resume
```

Notes:

- Kaggle writes checkpoints under `/kaggle/working/runs/<name>/weights/`.
- Download `best.pt` and `last.pt` from notebook output before session ends.
- If you need to continue in a new session, upload `last.pt` as a Kaggle Dataset and place it at `/kaggle/working/runs/<name>/weights/last.pt` before running `--resume`.

---

## Local Long-Run Resume (macOS)

If a local training run is interrupted intentionally (for example before closing a laptop),
resume from the newest checkpoint in `runs/segment/runs/.../weights/last.pt`.

Example resume command used in this project:

```bash
cd "/Users/tadun/Documents/2026/Final Project/medtube_segmentation"
rs_env/bin/python -c "from ultralytics import YOLO; YOLO('runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/last.pt').train(resume=True)"
```

Tip for long local runs:

```bash
mkdir -p runs/overnight_logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="runs/overnight_logs/train_compare_${TS}.log"
caffeinate -dimsu rs_env/bin/python train_compare.py --data "/Users/tadun/Downloads/MedTube Segmentation.yolov8 (1)/data.yaml" --aug-preset mild 2>&1 | tee "$LOG"
```

---

## Depth Experiment Plan

Depth-only training is useful as an ablation baseline, but not expected to be the final best model.

Recommended order:

1. RGB-only baseline.
2. Depth-only baseline.
3. RGB-D fusion model.

Notes:

- Model version alone (YOLOv8/9/11) is usually less important than depth quality and fusion strategy.
- Use paired and aligned depth when possible; colorized preview depth can reduce depth-only usefulness.

### Run output

```text
runs/
  YYYY-MM-DD_HH-MM-SS/
    YOLOv8-seg/weights/best.pt
    YOLOv9-seg/weights/best.pt
    YOLOv11-seg/weights/best.pt
    comparison.json
```

---

## Project Structure

```text
medtube_segmentation/
  collect_dataset.py    Dataset collection
  realsense_stream.py   Live preview
  train_compare.py      YOLO model comparison training
  kaggle_train.py       Kaggle single-model trainer
  docs/                 Project notes and report material
  tools/                QA and visualization utilities
  annotation/           Staged images + manifest for annotation
  data/                 Annotated dataset — YOLO format  (not committed)
  dataset/              Raw collected frames             (not committed)
  rs_env/               Python virtual environment       (not committed)
```
