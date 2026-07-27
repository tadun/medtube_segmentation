# MedTube Segmentation

Real-time instance segmentation of medical tubes by cap type, developed as the upstream vision component of an AI-enabled waste-sorting pipeline. Uses an Intel RealSense D415 depth camera and YOLO-family models across RGB, RGBD, and depth-only modalities.

**Dataset:** [Roboflow Universe](https://universe.roboflow.com/tades-workspace/medtube-segmentation)  
**Report:** `report/main.tex`

---

## Results summary

| Model | Input | Params | Box mAP₅₀₋₉₅ | Mask mAP₅₀₋₉₅ | ms/img |
| --- | --- | --- | --- | --- | --- |
| YOLO11n-RGBD | RGBD | 2.8 M | 0.988 | **0.929** | 22.4 |
| YOLOv8m-seg | RGB | 27.2 M | 0.968 | 0.905 | 108.5 |
| YOLO26n-seg | RGB | 2.7 M | 0.951 | 0.820 | 23.1 |
| YOLO11n-seg | RGB | 2.8 M | 0.949 | 0.820 | 21.7 |
| YOLO26n-depth | Depth | 2.7 M | 0.919 | 0.804 | 22.8 |

Evaluated on a 450-image held-out test split. Inference on Apple M1 Max CPU.

---

## Repository structure

```text
medtube_segmentation/
├── src/
│   ├── capture_dataset.py      # RealSense D415 data collection
│   ├── realsense_stream.py     # Live inference — 2×2 RGB+Depth display
│   ├── stream_multimodel.py    # Side-by-side multi-model comparison stream
│   ├── prepare_depth_dataset.py# Build RGBD / depth-only dataset variants
│   ├── train_compare.py        # Local multi-model training runner
│   └── train_kaggle.py         # Cloud/Kaggle training with checkpoint resume
├── tools/
│   ├── generate_report_figures.py  # Generate dataset grid + comparison chart
│   ├── eval_comparison.py          # Evaluate all models and log metrics
│   ├── compare_confusion_matrices.py
│   ├── split_rgbd.py
│   ├── rebalance_classes.py
│   ├── preview_labels.py
│   ├── view_masks.py
│   └── backup_dataset.sh
├── weights/                    # Fine-tuned model weights for all five models
│   ├── yolov8m_seg.pt          # YOLOv8m-seg — RGB (0.905 mask mAP)
│   ├── yolo11n_seg.pt          # YOLO11n-seg — RGB nano
│   ├── yolo26n.pt              # YOLO26n-seg — RGB nano, best speed/accuracy
│   ├── yolo11n_rgbd.pt         # YOLO11n-RGBD — depth fusion (0.929 mask mAP)
│   └── yolo26n_depth-2.pt      # YOLO26n-depth — depth-only (0.804 mask mAP)
├── report/
│   ├── main.tex                # Project report (LaTeX)
│   ├── references.bib
│   ├── abstract.tex
│   ├── Academic.cls
│   └── figures/                # Generated and captured figures
├── stream.sh                   # Passwordless RealSense launcher (macOS)
└── .gitignore
```

Large directories are gitignored and downloaded locally:

| Directory | Contents | Source |
| --- | --- | --- |
| `dataset/` | 30 × 100 raw RGB+depth frames | Captured locally |
| `balanced_yolo/` | Class-balanced dataset variant | Roboflow export |
| `depth_yolo/` | Depth-only dataset | `src/prepare_depth_dataset.py` |
| `rgbd_split/` | RGBD dataset with 70/15/15 split | `tools/split_rgbd.py` |
| `runs/` | Training outputs and inference snapshots | Generated locally |

---

## Setup

```bash
python3.12 -m venv rs_env
source rs_env/bin/activate
pip install pyrealsense2 opencv-python numpy ultralytics roboflow matplotlib pillow
```

> **macOS:** `pyrealsense2` requires elevated privileges to claim the USB interface.  
> Run `./stream.sh` for passwordless launch (see the script for one-time setup).

---

## Key scripts

### Live inference stream

```bash
./stream.sh
# or: sudo rs_env/bin/python src/realsense_stream.py
```

Displays a 2×2 grid: raw RGB · RGB+masks · depth heatmap · depth+masks.

| Key | Action |
| --- | --- |
| `M` | Cycle between loaded models |
| `Space` | Save snapshot (4 panels) |
| `R` | Toggle recording at 2 fps |
| `Q` | Quit |

### Data collection

```bash
sudo rs_env/bin/python src/capture_dataset.py
```

Saves paired RGB PNGs and 16-bit depth PNGs to `dataset/tube_<N>/`.

### Evaluate all models

```bash
rs_env/bin/python tools/eval_comparison.py
```

### Regenerate report figures

```bash
rs_env/bin/python tools/generate_report_figures.py
```

Outputs `fig_dataset_examples.png` and `fig_model_comparison.png` to `report/figures/`.

---

## Class definitions

| Class | Cap type | Recycling path |
| --- | --- | --- |
| Push-on | 16 mm push-fit disc | Snap removal |
| Universal | 31 mm wide screw-cap | Unscrew |
| Screwcap | Narrow threaded (varied) | Unscrew |
| Other | Non-standard | Manual handling |

Classes are colour-coded in the live stream: Push-on = blue, Universal = red, Screwcap = green, Other = yellow.
