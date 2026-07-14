# MedTube Segmentation Project Report Notes

## 1. Project Scope

- Goal: instance segmentation of medical tubes using RGB images collected from Intel RealSense D415.
- Core workflow: collect data -> stage and upload -> annotate in Roboflow (SAM-assisted/manual) -> export YOLOv8 segmentation -> train and compare YOLOv8/YOLOv9/YOLOv11.
- Primary codebase root: `medtube_segmentation`.

## 2. Hardware and Environment

- Machine used for training and development: Apple M1 Max (arm64), 64 GB RAM.
- Python environment: `rs_env`.
- Torch environment check:
  - torch 2.12.0
  - MPS built: True
  - MPS available: True
- Ultralytics version observed: 8.4.56.

## 3. Dataset and Taxonomy

- Initial collection stage completed for tube folders under `dataset/`.
- Staging pipeline created to aggregate RGB images for annotation.
- Class mapping in current staging logic:
  - Universal: tubes 2, 3, 11
  - Screw cap: tube 1
  - Push-on: tubes 4, 5, 6, 7, 12, 13, 14
  - Other: tubes 8, 9, 10, 15, 16, 17
- Files updated for taxonomy and staging include:
  - `prepare_annotation_set.py`
  - `README.md`

## 4. Annotation and Export Workflow

- Images staged for Roboflow annotation and uploaded.
- Roboflow annotation completed and YOLOv8 segmentation export downloaded.
- Export folder in use during training/debug:
  - `/Users/tadun/Downloads/MedTube Segmentation.yolov8 (1)`
- For project cleanliness, local export folder in repo is ignored via `.gitignore`.

## 5. Data Integrity Findings and Fixes

### 5.1 Empty labels

- Empty label files were detected in the YOLO export.
- Option 3 (copy nearest frame label) was attempted for 15 files.
- Visual QA showed copied labels had correct size but wrong placement.
- Dataset was restored from backup to pre-option-3 state for those files.
- A correction queue CSV was generated:
  - `MedTube Segmentation.yolov8/correction_queue_empty_labels.csv`

### 5.2 Training crash root cause

- Training failed with tensor size mismatch and mixed detect/segment warning.
- Root cause: 4 label rows were bounding-box format (5 values) inside segmentation label files.
- Quarantine action performed:
  - Removed 4 bad image/label pairs from active train/test splits.
  - Moved to:
    - `/Users/tadun/Downloads/MedTube Segmentation.yolov8 (1)/excluded_bad_seg_rows`
- Post-fix split counts:
  - train: 2097 images, 2097 labels
  - valid: 450 images, 450 labels
  - test: 449 images, 449 labels
- Remaining malformed short segmentation rows after quarantine: 0.

## 6. Training Pipeline Status

- Comparison script: `train_compare.py`.
- Models configured:
  - YOLOv8-seg: yolov8m-seg.pt
  - YOLOv9-seg: updated to yolov9c-seg.pt (valid checkpoint)
  - YOLOv11-seg: yolo11m-seg.pt
- Augmentation preset used for most runs: mild.

### 6.1 Smoke Test (Completed)

- Test run: YOLOv8-seg, 1 epoch, imgsz 640, batch 8.
- Dataset path used:
  - `/Users/tadun/Downloads/MedTube Segmentation.yolov8 (1)/data.yaml`
- Result summary (epoch 1 validation):
  - Box mAP50: 0.836
  - Box mAP50-95: 0.704
  - Mask mAP50: 0.836
  - Mask mAP50-95: 0.686
- Smoke test confirms training pipeline now runs end-to-end after label cleanup.

## 7. Utility and QA Scripts Added

- `tools/preview_autofilled_labels.py`
  - Generates overlays for auto-filled labels to inspect geometric correctness.
- `tools/view_coco_masks.py`
  - Local viewer for COCO segmentation overlays with class names.

## 8. Git History Milestones (Recent)

- ba23507: ignore MedTube export folder and add QA preview utility
- 1677cc1: collect UX improvements (tube switching, warning banner, etc.)
- e6806e5: style and linting cleanup (pylint 10.00/10 cycle)
- 5429b52: pre-commit setup and formatting
- b3b85d3: taxonomy update for Screw cap
- de6ec3c: tube class mapping in annotation manifest
- 1931a16: SAM3 + Roboflow workflow and preflight additions

## 9. Current Known Caveats

- Some labels in exported datasets required manual correction due to empties/malformed rows.
- YOLO training should use a cleaned segmentation-only dataset.
- If using depth in future, current pipeline remains RGB-only unless a custom RGB-D loader/model path is implemented.

## 10. Recommended Next Steps for Final Report Figures/Tables

- Add a dataset table with per-split image counts after cleaning.
- Add a model comparison table (YOLOv8/9/11) using the same cleaned dataset and same augmentation preset.
- Include a short error-analysis section on malformed labels and remediation process.
- Include one qualitative figure with predicted masks per class.

## 11. Reproducibility Checklist

- Verify `data.yaml` path exists and points to cleaned dataset.
- Run a 1-epoch smoke test first.
- Run full `train_compare.py` on same split and preset.
- Archive `runs/` output and `comparison.json` for report appendix.

## 12. Session Update (2026-07-13)

### 12.1 Operational training decision (laptop lid constraint)

- User selected safe stop/resume workflow for active local training.
- Active run was interrupted intentionally with Ctrl+C (exit code 130).
- Latest checkpoint identified for resume:
  - `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/last.pt`
- Resume command validated and recorded:
  - `rs_env/bin/python -c "from ultralytics import YOLO; YOLO('runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/last.pt').train(resume=True)"`

### 12.2 Model comparison rationale (YOLOv8/9/11)

- Comparison retained to benchmark across stable (v8), intermediate/newer (v9), and latest-generation (v11) segmentation families under identical data and augmentation settings.
- This isolates architecture generation effects from dataset/pipeline effects.

### 12.3 Depth-direction conclusion

- Depth-only training is considered worthwhile as an ablation baseline.
- Expected ranking for final accuracy remains:
  - RGB-D fusion (best expected)
  - RGB-only or depth-only depending on scene conditions
- Practical guidance captured for next experiments:
  - run RGB-only baseline,
  - run depth-only baseline,
  - then run RGB-D fusion,
  - compare mAP and class-wise failure modes.

## 13. Session Update (2026-07-14)

### 13.1 Training status

- YOLOv8-seg training is actively running (resumed from `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/last.pt`).
- Current working weights for live inference: `weights.pt` (project root symlink/copy).
- When training completes, swap `weights.pt` → new `best.pt` to update the live stream.

### 13.2 Live stream overhaul (`realsense_stream.py`)

Complete rewrite of the stream script into a production-ready YOLO segmentation viewer:

#### Display

- 2×2 grid window ("MedTube Segmentation Stream") auto-sized to fit the screen.
- TL: raw RGB stream, TR: depth heatmap (auto-ranged TURBO), BL: stream + YOLO masks, BR: depth + YOLO masks.
- Class-coloured masks matching data.yaml class order: Universal=red, Screwcap=green, Push-on=blue, Other=yellow.
- Panel labels drawn with `FONT_HERSHEY_DUPLEX` on semi-transparent dark backgrounds.
- HUD bar shows live/rec status, elapsed time, snapshot count, and full-text controls.

#### Capture

- Space saves a 4-view snapshot set to `runs/captures/snapshots/`.
- R starts/stops continuous recording to `runs/captures/rec_<timestamp>/` at 0.5 s intervals.

#### Camera/depth handling

- 180° flip applied at source (camera is mounted upside-down).
- 4 s auto-exposure warmup after pipeline start.
- Depth ROI locked on first stable frame to remove IR parallax zone.
- Depth post-processing: spatial filter → temporal filter → hole-filling filter.
- Depth+Masks panel uses scene-calibrated 435–535 mm range (matches collect_dataset.py).
- Heatmap panel uses per-frame auto-range for full depth scene visibility.
- Mask alignment uses `masks.xy` polygon rasterization (no interpolation offset).

### 13.3 Passwordless launcher (`run_rs.sh`)

- One-time setup creates `/usr/local/bin/rs-stream-medtube` and a scoped NOPASSWD sudoers rule.
- After setup: `./run_rs.sh` launches the stream without a password prompt.
- Scoped to the specific binary only — minimal security exposure.

### 13.4 macOS USB access root cause

- macOS CoreMediaIO / VDCAssistant claims the camera USB interface, blocking librealsense.
- Google Meet / video calls in particular hold the interface.
- Solution: run via `sudo` (handled by `run_rs.sh`).
- Camera connected through a USB 3.1 GenesysLogic hub — works reliably after sudo.

### 13.5 Current known issues

- Screwcap / Push-on confusion occurs occasionally — a model accuracy issue expected to improve once the current training run completes.
- Dark matte surface absorbs IR — depth heatmap has scattered black holes on that surface type; spatial/temporal/hole-filling filters reduce but do not eliminate this.

## 14. Session Update (2026-07-14 — Training Complete)

### 14.1 YOLOv8m-seg Full Run — Training Summary

- Run ID: `2026-07-12_22-48-54`
- Weights directory: `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/`
- Total epochs: **100** (completed; no early stop — patience=20 not triggered)
- Training device: Apple M1 Max CPU (MPS was not used due to Ultralytics CPU-fallback)
- Base model: `yolov8m-seg.pt` (pretrained COCO)
- Dataset: `/Users/tadun/Downloads/MedTube Segmentation.yolov8 (1)/data.yaml`
  - train: 2097 images, valid: 450 images, test: 449 images
- Batch size: 8, imgsz: 640

**Best checkpoint: epoch 85** (saved as `best.pt`, selected by Mask mAP50-95)

| Metric | Value |
| --- | --- |
| Box Precision | 0.9961 |
| Box Recall | 0.9981 |
| Box mAP50 | 0.9941 |
| Box mAP50-95 | 0.9730 |
| Mask Precision | 0.9961 |
| Mask Recall | 0.9981 |
| Mask mAP50 | 0.9941 |
| Mask mAP50-95 | **0.9107** |

**Final epoch (100) — validation split:**

| Metric | Value |
| --- | --- |
| Box mAP50 | 0.9941 |
| Box mAP50-95 | 0.9763 |
| Mask mAP50 | 0.9941 |
| Mask mAP50-95 | 0.9077 |

### 14.2 Test-Split Evaluation (best.pt on held-out test set)

- Evaluated with `model.val(split='test', imgsz=640, batch=8, device='cpu')`
- Results saved to: `runs/segment/runs/test_results/YOLOv8-best-test/`
- 449 images, 0 backgrounds, 0 corrupt

**Per-class results (test split):**

| Class | Images | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **all** | 449 | 0.990 | 0.991 | 0.985 | 0.968 | 0.991 | 0.993 | 0.985 | 0.806 |
| Other | 144 | 0.996 | 0.993 | 0.995 | 0.957 | 1.000 | 1.000 | 0.995 | 0.748 |
| Push-on | 131 | 0.999 | 0.992 | 0.995 | 0.978 | 0.999 | 0.992 | 0.995 | 0.791 |
| Screwcap | 97 | 0.969 | 0.979 | 0.954 | 0.945 | 0.968 | 0.979 | 0.954 | 0.798 |
| Universal | 77 | 0.996 | 1.000 | 0.995 | 0.993 | 0.995 | 1.000 | 0.995 | 0.887 |

**Observations:**

- Screwcap is the hardest class (lowest mAP50-95 at 0.945 box / 0.798 mask) — consistent with live-stream confusion noted in §13.5.
- Universal achieves perfect recall (1.0) and highest mask mAP50-95 (0.887).
- The gap between mAP50 (~0.985) and mAP50-95 (~0.806) for masks suggests the model localises tubes well but mask tightness degrades at higher IoU thresholds — likely due to annotation polygon coarseness rather than model failure.

### 14.3 Training Hyperparameters (full, for reproducibility)

```yaml
model: yolov8m-seg.pt (pretrained COCO, then resumed from last.pt)
epochs: 100
batch: 8
imgsz: 640
optimizer: auto  # resolved to AdamW
lr0: 0.01
lrf: 0.01        # cosine LR final factor
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
box: 7.5
cls: 0.5
dfl: 1.5
amp: true
close_mosaic: 10
overlap_mask: true
mask_ratio: 4
dropout: 0.0
patience: 20
# Augmentation
hsv_h: 0.01
hsv_s: 0.4
hsv_v: 0.2
degrees: 20.0
translate: 0.05
scale: 0.25
shear: 2.0
perspective: 0.0001
flipud: 0.0
fliplr: 0.5
mosaic: 0.2
erasing: 0.2
auto_augment: randaugment
```

### 14.4 Key Artefact Paths

| Artefact | Path |
| --- | --- |
| Best weights | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt` |
| Last weights | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/last.pt` |
| Training metrics CSV | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/results.csv` |
| Training args | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/args.yaml` |
| Test-split predictions JSON | `runs/segment/runs/test_results/YOLOv8-best-test/predictions.json` |
| Test-split plots | `runs/segment/runs/test_results/YOLOv8-best-test/` |
| Training curve plots | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/results.png` |
| Confusion matrix | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/confusion_matrix_normalized.png` |

### 14.5 Live Stream — Testing with best.pt

- `weights.pt` (repo root, 5.8 MB) is a different/older model — **not** the newly trained weights.
- `FALLBACK_WEIGHTS` in `realsense_stream.py` already points to `best.pt` automatically if `weights.pt` is absent.
- To run stream explicitly with new weights:

  ```bash
  ./run_rs.sh --weights runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt
  ```

- To make `best.pt` the permanent default, copy it to `weights.pt`:

  ```bash
  cp runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt weights.pt
  ```

## 15. Live Stream Comparison — YOLO11n vs YOLOv8m (2026-07-14)

### 15.1 Discovery

After testing `best.pt` (YOLOv8m-seg) on the live stream, it was observed to perform **visually worse** than `weights.pt` despite the latter having nominally lower val/test metrics. Investigation revealed:

- `weights.pt` is a fine-tuned **YOLO11n-seg** (4 medtube classes), origin: a prior training run (likely Kaggle GPU run — no local weights folder preserved for that session).
- The discrepancy is explained primarily by **inference speed**: YOLOv8m runs at ~112 ms/frame vs YOLO11n at ~21 ms/frame — a **5.4× difference** that severely impacts live stream smoothness.

### 15.2 Head-to-Head Test-Split Benchmark (449 images)

Both models evaluated with `model.val(split='test', imgsz=640, batch=8, device='cpu')`.

| Metric | YOLO11n-seg (`weights.pt`) | YOLOv8m-seg (`best.pt`) |
| --- | --- | --- |
| File size | 5.8 MB | 52 MB |
| Inference speed | **20.7 ms/frame** | 111.6 ms/frame |
| Box mAP50 | 0.983 | **0.985** |
| Box mAP50-95 | 0.950 | **0.968** |
| Mask mAP50 | 0.983 | **0.985** |
| Mask mAP50-95 | **0.819** | 0.806 |

**Per-class Mask mAP50 (test split):**

| Class | YOLO11n | YOLOv8m |
| --- | --- | --- |
| Other | 0.995 | **0.995** |
| Push-on | 0.995 | **0.995** |
| Screwcap | 0.947 | **0.954** |
| Universal | 0.995 | **0.995** |

### 15.3 Analysis

- **Metric parity**: Both models are essentially equivalent on the test split — differences are within ~1–2 pp.
- **Mask mAP50-95**: YOLO11n is +1.3 pp better than YOLOv8m on masks at high IoU thresholds, suggesting YOLO11's architectural improvements (C3k2, SPPF, C2PSA attention) produce tighter segmentation polygons on this data.
- **Live stream feel**: The 5.4× speed advantage of YOLO11n makes it significantly smoother and more responsive. At 21 ms/frame it can approach 30–48 fps on CPU; YOLOv8m at 112 ms/frame is capped at ~9 fps.
- **Generalisation**: The test set is from the same Roboflow export as training data. The live stream is real-world with different lighting and backgrounds — the nano model's lighter capacity may also reduce over-fitting to the annotation distribution.

### 15.4 Recommended Next Step — YOLO11m Full Run

The local YOLO11m run (`2026-07-12_22-07-43`) was interrupted before saving any weights. A proper comparison requires:

- YOLO11m-seg trained for 100 epochs on the same cleaned dataset with the same mild augmentation preset.
- This would isolate architecture generation (YOLO11 vs v8) from model size (nano vs medium).
- Expected outcome: YOLO11m should beat YOLOv8m on both metrics and speed (YOLO11m is ~20% faster than YOLOv8m at equivalent size).

**Interim decision**: keep `weights.pt` (YOLO11n) as the default for live streaming. Use `best.pt` (YOLOv8m) only for controlled benchmark comparison.

### 15.5 Updated Artefact Paths

| Artefact | Path |
| --- | --- |
| YOLO11n weights (live default) | `weights.pt` (repo root) |
| YOLO11n test-split plots | `runs/segment/runs/test_results/YOLO11n-best-test/` |
| YOLOv8m weights | `runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt` |
| YOLOv8m test-split plots | `runs/segment/runs/test_results/YOLOv8-best-test/` |

## 16. Roboflow Cloud-Trained Models (2026-07-14)

### 16.1 Decision

Two additional models trained via Roboflow cloud (GPU) to broaden the comparison:

1. **YOLOv8n-seg** — nano-tier v8 baseline; also provides a size-comparison data point within the v8 family
2. **RF-DETR (Small)** — transformer-based instance segmentation; architecturally distinct from the YOLO family

### 16.2 Training Configuration

| Setting | YOLOv8n-seg | RF-DETR-S |
| --- | --- | --- |
| Platform | Roboflow cloud (GPU) | Roboflow cloud (GPU) |
| Input resolution | 640×640 | **384×384** (RF-DETR recommended) |
| Dataset | Same Roboflow export | Same Roboflow export |
| Architecture type | CNN + NMS | Transformer, NMS-free |

Note: RF-DETR uses 384×384 because that is its recommended optimum (Roboflow guidance). All YOLO runs use 640×640.

### 16.3 Reporting Considerations

- **Hardware difference**: Roboflow GPU ≠ local M1 Max CPU — inference speed cannot be directly compared to local YOLO benchmarks.
- **Input resolution difference** for RF-DETR (384 vs 640) must be disclosed in the comparison table.
- Present cloud-trained results in clearly labelled rows, e.g. `YOLOv8n-seg (640², cloud)` and `RF-DETR-S (384², cloud)`.
- Accuracy metrics (mAP) are still comparable since they are evaluated on the same test split regardless of training hardware.

### 16.4 Results (fill in once training completes)

| Model | Train platform | imgsz | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n-seg | Roboflow cloud | 640 | — | — | — | — |
| RF-DETR-S | Roboflow cloud | 384 | — | — | — | — |
