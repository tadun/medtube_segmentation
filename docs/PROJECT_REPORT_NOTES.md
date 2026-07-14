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

**Display**
- 2×2 grid window ("MedTube Segmentation Stream") auto-sized to fit the screen.
- TL: raw RGB stream, TR: depth heatmap (auto-ranged TURBO), BL: stream + YOLO masks, BR: depth + YOLO masks.
- Class-coloured masks matching data.yaml class order: Universal=red, Screwcap=green, Push-on=blue, Other=yellow.
- Panel labels drawn with `FONT_HERSHEY_DUPLEX` on semi-transparent dark backgrounds.
- HUD bar shows live/rec status, elapsed time, snapshot count, and full-text controls.

**Capture**
- Space saves a 4-view snapshot set to `runs/captures/snapshots/`.
- R starts/stops continuous recording to `runs/captures/rec_<timestamp>/` at 0.5 s intervals.

**Camera/depth handling**
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
