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

**Output**

```
dataset/
  YYYY-MM-DD_HH-MM-SS_<label>/
    rgb/          Colour PNGs  (1280 × 720, BGR)
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

```
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

**Output**

```
runs/
  YYYY-MM-DD_HH-MM-SS/
    YOLOv8-seg/weights/best.pt
    YOLOv9-seg/weights/best.pt
    YOLOv11-seg/weights/best.pt
    comparison.json
```

---

## Project Structure

```
medtube_segmentation/
  collect_dataset.py    Dataset collection
  realsense_stream.py   Live preview
  train_compare.py      YOLO model comparison training
  data/                 Annotated dataset — YOLO format  (not committed)
  dataset/              Raw collected frames             (not committed)
  rs_env/               Python virtual environment       (not committed)
```
