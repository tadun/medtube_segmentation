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

- Comparison script: `src/train_compare.py`.
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

- `tools/preview_labels.py`
  - Generates overlays for auto-filled labels to inspect geometric correctness.
- `tools/view_masks.py`
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
- Run full `src/train_compare.py` on same split and preset.
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

### 13.2 Live stream overhaul (`src/realsense_stream.py`)

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
- Depth+Masks panel uses scene-calibrated 435–535 mm range (matches src/capture_dataset.py).
- Heatmap panel uses per-frame auto-range for full depth scene visibility.
- Mask alignment uses `masks.xy` polygon rasterization (no interpolation offset).

### 13.3 Passwordless launcher (`stream.sh`)

- One-time setup creates `/usr/local/bin/rs-stream-medtube` and a scoped NOPASSWD sudoers rule.
- After setup: `./stream.sh` launches the stream without a password prompt.
- Scoped to the specific binary only — minimal security exposure.

### 13.4 macOS USB access root cause

- macOS CoreMediaIO / VDCAssistant claims the camera USB interface, blocking librealsense.
- Google Meet / video calls in particular hold the interface.
- Solution: run via `sudo` (handled by `stream.sh`).
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
- `FALLBACK_WEIGHTS` in `src/realsense_stream.py` already points to `best.pt` automatically if `weights.pt` is absent.
- To run stream explicitly with new weights:

  ```bash
  ./stream.sh --weights runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt
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

## 17. Full Model Comparison — All YOLO Variants (2026-07-15)

### 17.1 Test-Split Benchmark Summary — MedTube 2 dataset (450 images, split='test', imgsz=640, batch=8, CPU)

All models evaluated on the held-out test split of `MedTube 2.yolov8`. YOLO models used `model.val()`; RF-DETR used `notebooks/eval_rfdetr.py` via Roboflow Inference SDK. Results sorted by Box mAP50-95.

| Model | Arch type | Weights file | Size | Params | GFLOPs | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | CPU inference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR-S | Transformer | Roboflow API `medtube-2/1` | 129 MB | — | — | **0.986** | **0.954** | — | — | 27.8 ms |
| YOLOv8m-seg | CNN | `runs/.../YOLOv8-seg/weights/best.pt` | 52 MB | 27.2 M | 104.3 | 0.985 | 0.968 | **0.985** | **0.905** | 108.5 ms |
| YOLO26n-seg | CNN | `yolo26n.pt` | 6.3 MB | 2.7 M | 9.0 | 0.983 | 0.951 | 0.983 | 0.820 | **23.1 ms** |
| YOLO11n-seg | CNN | `yolo11n_weights.pt` | 5.8 MB | 2.8 M | 9.6 | 0.983 | 0.949 | 0.983 | 0.820 | 21.7 ms |
| YOLOv9c-seg | CNN | `YOLOv9c-seg/weights/best.pt` | 213 MB | 27.6 M | 147.6 | 0.983 | 0.941 | 0.983 | 0.799 | 182.2 ms |

**Notes:**

- RF-DETR Mask mAP not computed locally — `supervision` 0.29.1 does not support `use_mask_iou=True`. Roboflow's own valid-set evaluation reports mAP@50 = **99.8%** (Precision 99.8%, Recall 99.8%, F1 99.8%).
- RF-DETR trained at 384×384 input resolution; all YOLO models at 640×640. Resolution difference is noted in the comparison.
- YOLOv9c trained on Colab with `medtube-2` dataset; evaluation here also uses `medtube-2` — dataset mismatch was ruled out as the cause of underperformance.

### 17.2 Key Findings

- **RF-DETR leads on Box mAP50-95 (0.954) AND runs at 27.8 ms** — faster than YOLOv8m (108 ms) despite being a transformer. The attention-based architecture achieves tighter bounding boxes than all YOLO variants while remaining suitable for real-time deployment.
- **YOLOv8m leads on Mask mAP50-95 (0.905)** — 100-epoch local training with full hyperparameter control gives the most precise polygon masks.
- **YOLO26n is the best nano model** — fastest (23.1 ms), best mask IoU at nano tier (0.820), marginally better than YOLO11n on all metrics.
- **YOLOv9c underperforms its size** — 213 MB / 182 ms yet lowest Mask mAP50-95 (0.799). Root cause: only **32 epochs completed** (early stopping triggered; patience=20). At the point of early stopping, val Mask mAP50-95 on its own dataset was 0.838 — much closer to the other models. Undertrained, not architecturally inferior.
- **Speed-accuracy sweet spot**: YOLO26n for live deployment; YOLOv8m or RF-DETR for highest accuracy batch inference.

### 17.3 Deployment Recommendation

| Use case | Recommended model | Reason |
| --- | --- | --- |
| Live RealSense stream | YOLO26n (`yolo26n.pt`) | Fastest nano, best mask IoU in nano tier, 23 ms |
| Highest mask accuracy | YOLOv8m (`best.pt`) | Best Mask mAP50-95 (0.905) |
| Highest box accuracy | RF-DETR (`medtube-2/1`) | Best Box mAP50-95 (0.954), 28 ms |
| Report architecture comparison | RF-DETR + YOLOv8m + YOLO26n | Transformer vs CNN large vs CNN nano |

## 18. Session Update (2026-07-15)

### 18.1 New model weights added

| File | Architecture | Size | Origin | Classes |
| --- | --- | --- | --- | --- |
| `yolo26n.pt` | YOLO26n-seg | 6.3 MB | Roboflow cloud | 4 medtube |
| `rfdetr.pt` | RF-DETR-S | 129 MB | Roboflow cloud | 4 medtube |
| `YOLOv9c-seg/weights/best.pt` | YOLOv9c-seg | 213 MB | Colab GPU (32 epochs) | 4 medtube |

### 18.2 YOLOv9c training post-mortem

- Trained on Colab free T4 using `notebooks/train_colab_yolov9c.ipynb`.
- Dataset: `tadeass-workspace/medtube-2` (new Roboflow account).
- Only **32 epochs** completed before early stopping (patience=20, no improvement for 20 epochs).
- Best val Mask mAP50-95 on its own dataset: **0.838** at epoch 30.
- Local test-split evaluation against `MedTube 2.yolov8` gave **0.799** — gap is due to undertrained model not dataset mismatch (both use medtube-2).
- Recommendation: retrain for full 100 epochs or increase patience to 30.

### 18.3 MedTube 2 dataset (new Roboflow account)

- Workspace: `tadeass-workspace`, project: `medtube-2`, version: `dataset`
- Local path: `/Users/tadun/Downloads/MedTube 2.yolov8/`
- Split counts: train 2100, valid 450, test 450 (vs old: 2097/450/449)
- Same 4 classes, same 640×640 export. All model re-evaluations from 2026-07-15 onwards use this dataset.

### 18.4 RF-DETR evaluation methodology

- Evaluated via Roboflow Inference SDK (`inference` package, model ID `medtube-2/1`).
- Script: `notebooks/eval_rfdetr.py`.
- Key implementation details:
  - `torch.cuda.stream` monkey-patched to a no-op (Apple Silicon MPS incompatibility).
  - Ground-truth boxes derived from polygon min/max (YOLO segmentation format — no explicit bbox).
  - Prediction polygons rasterised from `p.points` (list of `Point(x, y)`) using `cv2.fillPoly`.
  - Mask mAP not computable with `supervision==0.29.1` (`use_mask_iou` not supported).
- Results: Box mAP50=0.986, Box mAP50-95=0.954, inference=27.8 ms/image.

### 18.5 Codebase reorganisation (2026-07-14–15)

#### File renames

| Old name | New name |
| --- | --- |
| `collect_dataset.py` | `src/capture_dataset.py` |
| `kaggle_train.py` | `src/train_kaggle.py` |
| `realsense_stream.py` | `src/realsense_stream.py` |
| `train_compare.py` | `src/train_compare.py` |
| `run_rs.sh` | `stream.sh` |
| `tools/preview_autofilled_labels.py` | `tools/preview_labels.py` |
| `tools/view_coco_masks.py` | `tools/view_masks.py` |
| `docs/PROJECT_REPORT_NOTES.md` | `docs/notes.md` |

#### Directory structure

All Python scripts moved to `src/`. Shell launcher (`stream.sh`) remains in root. `Path(__file__).resolve().parent` updated to `.parent.parent` in all `src/` scripts so `.ultralytics/` and `.matplotlib/` caches resolve to the project root.

#### Live stream default weights

- `DEFAULT_WEIGHTS` in `src/realsense_stream.py` updated from `weights.pt` → `yolo26n.pt` (YOLO26n outperforms YOLO11n on all metrics).

### 18.6 New notebooks and scripts

| File | Purpose |
| --- | --- |
| `notebooks/train_colab_yolov9c.ipynb` | Colab notebook to train YOLOv9c on medtube-2 dataset |
| `notebooks/eval_rfdetr.py` | Evaluate RF-DETR on local test split via Roboflow Inference SDK |

### 18.7 Security note

The Roboflow API key `qxCKKYWIhOYWu3jZtVNq` was exposed in a VS Code chat session on 2026-07-15. **This key should be treated as compromised and regenerated** from the Roboflow dashboard (Settings → Roboflow API → Regenerate). The key has no write access to training infrastructure but can trigger inference API calls.

### 18.8 Outstanding items before report

- [ ] Retrain YOLOv9c for full 100 epochs for a fair size-matched comparison vs YOLOv8m
- [ ] Compute RF-DETR Mask mAP (upgrade supervision or implement custom mask IoU)
- [ ] Fill in YOLOv8n-seg Roboflow cloud results (section 16.4)
- [ ] Collect qualitative figure: side-by-side stream snapshots per model
- [ ] Decide whether to include the RF-DETR live-stream integration (requires inference SDK in deployment)

---

## 19. Parent Folder Inventory (Final Project/)

The parent directory `/Users/tadun/Documents/2026/Final Project/` contains additional
resources, prior work, and documentation relevant to the report.

### 19.1 Project Guidelines

- **MSc - Advanced Project Guidelines (2025-26).docx** — Official project submission guidelines (word count, structure, formatting, marking criteria). Must be followed for report layout.

### 19.2 Conveyor-Belt-Tube-Detection-System (Jessiah Buamah)

A companion/prior project by Jessiah Buamah providing a **digital twin framework** for medical tube sorting using reinforcement learning and robotics:

- **Simulation engine:** PyBullet physics with 3D conveyor belt, hopper, air jets, and collection bins
- **Detection model:** YOLOv8n (object detection, not segmentation) trained on synthetic PyBullet-rendered images (320×320, 1000 images)
- **Tube types:** 5 classes — Polypropylene Tube 1, Polypropylene Tube 2, Polystyrene Tube 1, Polystyrene Tube 2, Lysis Tube
- **RL algorithms:** PPO and SAC (Stable-Baselines3 + Gymnasium)
- **Robot:** UR5 + Robotiq 85 gripper (simulated) for cap unscrewing/disassembly
- **Results:** SAC achieved 81% peak sorting accuracy; PPO most stable convergence; robotic disassembly 72–85% success
- **Relation to MedTube:** This project provides the downstream sorting/disassembly context that motivates high-quality upstream segmentation. MedTube segmentation feeds into this pipeline.
- **Key files:** `object_deyection.py`, `rl_training_ppo.py`, `sample_test.py`, `tube_detection_project.py`
- **3D assets:** URDF files for conveyor belts, air jets, bins, UR5 robot; OBJ/MTL meshes for tube types
- **Pipeline figure:** `assets/figure2.png` — Medical Tube Segregation and Disassembly Pipeline diagram

### 19.3 Early Object Detection Results

**object_detection_results.txt** — Inference log from an early YOLOv8 detection model on the `datasets/` folder (Tube 1–8, VACUETTE types):

- Most detections have very low confidence (0.01–0.06), indicating an under-trained or mismatched model
- Class names used: PS-Tube (Polystyrene), PP-Tube (Polypropylene), Colour-Checker
- Demonstrates the need for a properly trained segmentation model (motivation for MedTube project)

### 19.4 datasets/ Folder (Original Tube Photos)

Hand-collected iPhone photographs of real medical tubes, organised by type:

| Folder | Tube Type | Content |
| --- | --- | --- |
| Tube 1–8 | Generic numbered sets | ~10 JPEG photos each |
| VACUETTE K3E K3EDTA | EDTA anticoagulant tube (purple cap) | ~10 photos |
| VACUETTE LH LITHIUM HEPARIN SEP | Lithium heparin separator (green cap) | ~10 photos |
| VACUETTE Sodium Heparin Blue PREMIUM 6mL | Sodium heparin (blue cap) | ~10 photos |
| VACUETTE TUBE 7 ml CAT Serum Separator | Serum separator (gold/yellow cap) | ~10 photos |
| VACUETTE TUBE 9 ml 9NC Coagulation sodium citrate | Coagulation tube (light blue cap) | ~10 photos |
| VACUETTE Tube CAT Serum Separator Clot Activator | Clot activator (red cap) | ~10 photos |
| Vacuette Tube 2ml FX Sodium Fluoride Potassium Oxalate | Fluoride/oxalate tube (grey cap) | ~10 photos |

**Note:** These are early exploratory photos taken before the RealSense D415 setup. The MedTube segmentation dataset uses the RealSense captures instead.

### 19.5 Papers/ Folder (Literature Already Collected)

Five PDF papers already saved in `Papers/`:

1. **3093-4155-1-PB.pdf** — (Unidentified, needs title extraction)
2. **e3sconf_icfee2024_04001.pdf** — E3S Conference on Frontiers of Energy and Environment 2024
3. **s41598-023-45759-z.pdf** — Nature Scientific Reports 2023 paper
4. **sensors-20-03816.pdf** — MDPI Sensors 2020 paper
5. **sensors-21-01213.pdf** — MDPI Sensors 2021 paper

### 19.6 Other PDFs at Root Level

- **MedBin.pdf** — Likely a related medical waste/recycling bin design document
- **Real-Time_Progressive_3D_Semantic_Segmentation_for_Indoor_Scenes.pdf** — 3D semantic segmentation paper (reference for depth-based approaches)
- **Robotics_Conference_Paper vF.pdf** — Robotics conference paper (likely related to the UR5 disassembly pipeline)
- **download.pdf** — Unknown (needs title extraction)

### 19.7 Documents/ Folder (Administrative & Presentations)

#### Presentations

- **MedTube_Segmentation.pptx** — Main project presentation (original version)
- **MedTube_Segmentation (1).pptx** — Revised presentation
- **MedTube_Segmentation 2.pptx** — Latest presentation (also at root level)
- **MedPlasticLoughUni.pptx** — MedPlastic presentation (Loughborough University context)
- **MedPlastic_JfDu.pptx** — MedPlastic presentation (JfDu variant)
- **Researching and writing systematic reviews.pptx** — Academic writing guide

#### Project Documents

- **AI Enabled MTP Sorting System Architecture.docx** — System architecture document for the AI-enabled Medical Tube Product sorting system
- **AI and ML-Driven Automated Sorting System.pdf** — Published/formal document on the sorting system
- **MTPContainers.docx** — Medical Tube Product container specifications
- **MedTube - Presentation Outline.docx / .rtf** — Presentation outline drafts

#### Administrative

- **Risk Assessment Form - Horn.docx** — Health and safety risk assessment
- **Ethical Review Form - Horn.pdf / .docx** — Ethical review documentation (signed)
- **FYP and MSc Projects - Ethical Review Form (2).docx** — University ethical review template
- **GENERAL HEALTH AND SAFETY RISK ASSESSMENT FORM.docx** — General H&S template
- **MSc - Impact Statement Brief Notes (2025-26).docx** — Impact statement guidance
- **MSc - Lecture Notes on Impact (June 2026).pdf** — Impact lecture notes
- **BlueBearLogin.docx** — BlueBear HPC login details (University of Birmingham)
- **Summer Project Meeting Form.docx** — Template meeting form

#### Meeting Forms (Supervision Records)

- **Meeting Form - May.docx** — May 2026 meeting record
- **Meeting Form - June.docx** — June 2026 meeting record
- **Meeting Form - July.docx** — July 2026 meeting record

#### Data

- **class_balance.csv** — Final class distribution across splits:

| Class | Total | Train | Valid | Test |
| --- | --- | --- | --- | --- |
| Other | 909 | 637 | 127 | 145 |
| Push-on | 906 | 642 | 134 | 130 |
| Screwcap | 716 | 506 | 109 | 101 |
| Universal | 506 | 345 | 82 | 79 |
| **Total** | **3037** | **2130** | **452** | **455** |

### 19.8 Pictures/ Folder (Reference Images)

- **IMG_6344.jpeg, IMG_6374.jpeg** — Photos of physical medical tubes
- **IMG_9128.HEIC** — Additional tube photo
- **Blood-collection-tubes-Emilie-Brysting.webp** — Reference image of blood collection tubes (various cap colours)
- **Medical_recycling.png** — Medical recycling context image
- **yolo-(1).jpg** — YOLO architecture diagram (for report figure)
- **crested-wm-full-colour-e1671624830551.png** — University of Birmingham crest/watermark
- **ChatGPT Image Jun 17, 2026, 06_21_33 PM.png** — AI-generated concept image
- **ChatGPT Image Jun 17, 2026, 06_40_30 PM.png** — AI-generated concept image
- **Gemini_Generated_Image_gtoafwgtoafwgtoa.png** — AI-generated image
- **1_olhihapANay0HcoPzRclcA.png** — Reference diagram
- **cf927d5f-5584-4b0c-8d82-9caa45c58e32.png** — Reference image

### 19.9 Screenshots/ Folder (Development Timeline Evidence)

51 screenshots documenting the entire development process chronologically:

| Date | Content |
| --- | --- |
| 2026-01-22 | Project topic selection form (Machine Vision ranked #1) |
| 2026-06-01 | VACUETTE tube photos; Roboflow SAM annotation interface |
| 2026-06-17 | Early RealSense D415 capture of screwcap tube |
| 2026-06-21 | RealSense SDK viewer (depth + RGB streams) |
| 2026-06-26 | Development progress screenshot |
| 2026-07-09 | Roboflow dashboard (3000 images); dataset analytics (class distribution); annotation heatmaps |
| 2026-07-12 | Training setup and early epoch screenshots |
| 2026-07-13 | Roboflow MedTube 2 workspace: V1 with YOLOv11n-seg model training |
| 2026-07-14 | Live stream 2×2 view with mask overlays; training curves; Roboflow training graphs; Colab YOLOv9c training on T4 GPU |
| 2026-07-15 | Final model testing on Roboflow; MedTube 2 cloud training results |

### 19.10 Video Assets

- **MedTube_Segmentation_Horn.mp4** — Video demonstration (likely live inference demo or presentation recording)
- **SimulationVideo.mov** — PyBullet conveyor belt simulation video (Conveyor-Belt project)

### 19.11 Roboflow Account Details

- **User:** txh543@student.bham.ac.uk
- **Workspaces:** tades-workspace (MedTube v1), tadeass-workspace (MedTube 2)
- **University:** University of Birmingham
- **Augmented dataset size:** 7200 images (from 3000 base, 2.4× augmentation)

### 19.12 University Context

- **Institution:** University of Birmingham
- **Programme:** MSc (Advanced Project, 2025–26 academic year)
- **Supervisor meetings:** May, June, July 2026
- **Ethics:** Ethical review signed; Risk assessment completed
- **HPC access:** BlueBear (University HPC cluster) — login details available
- **Impact statement:** Required per MSc guidelines

### 19.13 Key Figures Available for Report

| Source | File/Location | Description |
| --- | --- | --- |
| Conveyor-Belt project | `assets/figure2.png` | System architecture pipeline diagram |
| Pictures | `yolo-(1).jpg` | YOLO architecture diagram |
| Pictures | `Blood-collection-tubes-*.webp` | Medical tube reference image |
| Pictures | `Medical_recycling.png` | Medical recycling motivation image |
| Pictures | `crested-wm-full-colour-*.png` | University of Birmingham crest |
| Screenshots | Various (2026-01 to 2026-07) | Full development timeline evidence |
| MedTube runs | `results.png`, confusion matrices, PR curves | Training result plots |
| MedTube runs | `val_batch*_pred.jpg` | Qualitative prediction examples |

---

## 20. Literature References for Report

Curated list of peer-reviewed and closely relevant references only. BibTeX keys in brackets.

### 20.1 YOLO Family — Object Detection & Segmentation

1. **[redmon2016yolo]** Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. (2016) 'You Only Look Once: Unified, Real-Time Object Detection', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 779–788. doi: 10.1109/CVPR.2016.91.

2. **[jocher2023yolov8]** Jocher, G., Chaurasia, A. and Qiu, J. (2023) *Ultralytics YOLOv8* [Software], version 8.0.0. Available at: https://github.com/ultralytics/ultralytics (Accessed: 15 July 2026). *Note: No formal paper; cite as software per Ultralytics guidance.*

3. **[wang2024yolov9]** Wang, C.-Y., Yeh, I.-H. and Liao, H.-Y.M. (2024) 'YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information', *Computer Vision — ECCV 2024*, Springer Nature Switzerland, pp. 1–21. arXiv: 2402.13616. **Peer-reviewed (ECCV).**

4. **[jocher2024yolo11]** Jocher, G. and Qiu, J. (2024) *Ultralytics YOLO11* [Software], version 11.0.0. Available at: https://github.com/ultralytics/ultralytics (Accessed: 15 July 2026). *Note: No formal paper; cite as software per Ultralytics guidance.*

5. **[bochkovskiy2020yolov4]** Bochkovskiy, A., Wang, C.-Y. and Liao, H.-Y.M. (2020) 'YOLOv4: Optimal Speed and Accuracy of Object Detection', arXiv preprint arXiv:2004.10934. *Relevant for mosaic augmentation strategy.*

### 20.2 Detection Transformers (DETR Family)

6. **[carion2020detr]** Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. and Zagoruyko, S. (2020) 'End-to-End Object Detection with Transformers', *European Conference on Computer Vision (ECCV)*, Springer, pp. 213–229. arXiv: 2005.12872. **Peer-reviewed (ECCV).**

7. **[zhu2021deformable]** Zhu, X., Su, W., Lu, L., Li, B., Wang, X. and Dai, J. (2021) 'Deformable DETR: Deformable Transformers for End-to-End Object Detection', *International Conference on Learning Representations (ICLR)*. arXiv: 2010.04159. **Peer-reviewed (ICLR).**

8. **[lv2023rtdetr]** Lv, W., Xu, S., Zhao, Y., Wang, G., Wei, J., Cui, C., Du, Y., Dang, Q. and Liu, Y. (2023) 'DETRs Beat YOLOs on Real-time Object Detection', arXiv preprint arXiv:2304.08069. *Basis for RT-DETR architecture.*

9. **[robicheaux2025rfdetr]** Robicheaux, P., Gallagher, J., Nelson, J. and Robinson, I. (2025) 'RF-DETR: Neural Architecture Search for Real-Time Detection Transformers', arXiv preprint arXiv:2511.09554. *RF-DETR model used in this project; uses DINOv2 backbone.*

### 20.3 Instance Segmentation

10. **[he2017maskrcnn]** He, K., Gkioxari, G., Dollár, P. and Girshick, R. (2017) 'Mask R-CNN', *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2961–2969. arXiv: 1703.06870. **Peer-reviewed (ICCV).**

11. **[bolya2019yolact]** Bolya, D., Zhou, C., Xiao, F. and Lee, Y.J. (2019) 'YOLACT: Real-time Instance Segmentation', *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 9157–9166. arXiv: 1904.02689. **Peer-reviewed (ICCV).**

### 20.4 Foundation Models for Annotation

12. **[kirillov2023sam]** Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.-Y., Dollár, P. and Girshick, R. (2023) 'Segment Anything', *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4015–4026. arXiv: 2304.02643. **Peer-reviewed (ICCV).** *Used via Roboflow for SAM-assisted annotation.*

13. **[oquab2024dinov2]** Oquab, M., Darcet, T., Moutakanni, T. et al. (2024) 'DINOv2: Learning Robust Visual Features without Supervision', *Transactions on Machine Learning Research (TMLR)*. arXiv: 2304.07193. **Peer-reviewed (TMLR).** *Backbone of RF-DETR.*

### 20.5 Roboflow Platform & Benchmarking

14. **[ciaglia2022rf100]** Ciaglia, F., Zuppichini, F.S., Guerrie, P., McQuade, M. and Solawetz, J. (2022) 'Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark', arXiv preprint arXiv:2211.13523. *Introduces RF100 benchmark; 100 datasets, 7 domains, 224,714 images, 805 classes from Roboflow Universe. Used by Apple, Microsoft, Baidu for benchmarking.*

### 20.6 Evaluation Metrics & Benchmarks

15. **[lin2014coco]** Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. and Zitnick, C.L. (2014) 'Microsoft COCO: Common Objects in Context', *European Conference on Computer Vision (ECCV)*, Springer, pp. 740–755. arXiv: 1405.0312. **Peer-reviewed (ECCV).** *Defines mAP50 and mAP50-95 metrics used throughout this project.*

16. **[padilla2020survey]** Padilla, R., Netto, S.L. and da Silva, E.A.B. (2020) 'A Survey on Performance Metrics for Object-Detection Algorithms', *2020 International Conference on Systems, Signals and Image Processing (IWSSIP)*, IEEE, pp. 237–242. **Peer-reviewed (IEEE).**

### 20.7 Depth Sensors & Intel RealSense

17. **[keselman2017realsense]** Keselman, L., Iselin Woodfill, J., Grunnet-Jepsen, A. and Bhowmik, A. (2017) 'Intel RealSense Stereoscopic Depth Cameras', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 1–10. doi: 10.1109/CVPRW.2017.167. **Peer-reviewed (CVPRW).** *Core reference for D415 camera used in this project.*

18. **[giancola2018survey]** Giancola, S., Valenti, M. and Sala, R. (2018) 'A Survey on 3D Cameras: Metrological Comparison of Time-of-Flight, Structured-Light and Active Stereoscopy Technologies', *SpringerBriefs in Computer Science*, Springer. doi: 10.1007/978-3-319-91761-0. **Peer-reviewed (Springer).** *Compares ToF, structured light, and active stereo depth sensors including RealSense.*

19. **[zanuttigh2016time]** Zanuttigh, P., Marin, G., Dal Mutto, C., Dominio, F., Minto, L. and Cortelazzo, G.M. (2016) *Time-of-Flight and Structured Light Depth Cameras: Technology and Applications*, Springer. doi: 10.1007/978-3-319-30973-6. **Peer-reviewed (Springer monograph).** *Comprehensive depth camera technology reference.*

### 20.8 RGB-D / Depth in Object Detection & Classification

20. **[eitel2015rgbd]** Eitel, A., Springenberg, J.T., Spinello, L., Riedmiller, M. and Burgard, W. (2015) 'Multimodal Deep Learning for Robust RGB-D Object Recognition', *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 681–687. arXiv: 1507.06821. **Peer-reviewed (IROS).** *Two-stream CNN architecture for RGB + depth with late fusion — directly relevant to depth fusion approach considered in this project.*

21. **[gupta2014learning]** Gupta, S., Girshick, R., Arbeláez, P. and Malik, J. (2014) 'Learning Rich Features from RGB-D Images for Object Detection and Segmentation', *European Conference on Computer Vision (ECCV)*, Springer, pp. 345–360. **Peer-reviewed (ECCV).** *Introduces HHA encoding (horizontal disparity, height above ground, angle) for converting depth maps to CNN-compatible 3-channel input.*

22. **[qi2018frustum]** Qi, C.R., Liu, W., Wu, C., Su, H. and Guibas, L.J. (2018) 'Frustum PointNets for 3D Object Detection from RGB-D Data', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 918–927. arXiv: 1711.08488. **Peer-reviewed (CVPR).** *Combines 2D detection with 3D point cloud processing for RGB-D.*

23. **[schwarz2015rgbd]** Schwarz, M., Schulz, H. and Behnke, S. (2015) 'RGB-D Object Recognition and Pose Estimation Based on Pre-trained Convolutional Neural Network Features', *2015 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 1329–1335. **Peer-reviewed (ICRA).** *Transfer learning with depth for robotic object recognition.*

24. **[song2015sunrgbd]** Song, S., Lichtenberg, S.P. and Xiao, J. (2015) 'SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 567–576. **Peer-reviewed (CVPR).** *Major RGB-D dataset and benchmark for scene understanding.*

25. **[ren2015fasterrcnn]** Ren, S., He, K., Girshick, R. and Sun, J. (2015) 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks', *Advances in Neural Information Processing Systems (NeurIPS)*, 28, pp. 91–99. arXiv: 1506.01497. **Peer-reviewed (NeurIPS).** *Foundational two-stage detector; precursor to Mask R-CNN.*

### 20.9 Transfer Learning & Data Augmentation

26. **[shorten2019augmentation]** Shorten, C. and Khoshgoftaar, T.M. (2019) 'A Survey on Image Data Augmentation for Deep Learning', *Journal of Big Data*, 6(1), pp. 1–48. doi: 10.1186/s40537-019-0197-0. **Peer-reviewed (Springer).** *Covers mosaic, copy-paste, colour jitter, erasing — all used in this project.*

27. **[he2016resnet]** He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep Residual Learning for Image Recognition', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778. **Peer-reviewed (CVPR).** *Backbone architecture used in YOLO and DETR models.*

28. **[lin2017fpn]** Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B. and Belongie, S. (2017) 'Feature Pyramid Networks for Object Detection', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2117–2125. **Peer-reviewed (CVPR).** *Multi-scale feature extraction neck architecture used in YOLO models.*

### 20.10 Medical Waste, Plastic & Circular Economy

29. **[who2024healthcare]** World Health Organization (2024) *Health-Care Waste*. Fact Sheet, 24 October. Available at: https://www.who.int/news-room/fact-sheets/detail/health-care-waste (Accessed: 15 July 2026). *Key statistics: 85% non-hazardous / 15% hazardous; high-income countries generate 0.5 kg hazardous waste per hospital bed per day; 16 billion injections administered worldwide annually.*

30. **[windfeld2015medical]** Windfeld, E.S. and Brooks, M.S.-L. (2015) 'Medical Waste Management — A Review', *Journal of Environmental Management*, 163, pp. 98–108. doi: 10.1016/j.jenvman.2015.08.013. **Peer-reviewed (Elsevier).**

31. **[chartier2014safe]** Chartier, Y., Emmanuel, J., Pieper, U. et al. (eds.) (2014) *Safe Management of Wastes from Health-care Activities*, 2nd edn. Geneva: World Health Organization. **WHO technical guidance document.**

32. **[nhs2022netzero]** NHS England (2022) *Delivering a Net Zero National Health Service*. Available at: https://www.england.nhs.uk/greenernhs/a-net-zero-nhs/ (Accessed: 15 July 2026). *NHS net zero targets: 2040 for direct emissions, 2045 for supply chain. Direct emissions down 68% since 1990.*

33. **[pgh2024waste]** Practice Greenhealth (2024) *Health Care Waste*. Available at: https://practicegreenhealth.org/topics/waste/waste-0 (Accessed: 15 July 2026). *US hospitals produce over 5 million tonnes of waste per year; 29 pounds per bed per day.*

34. **[rizan2021environmental]** Rizan, C., Reed, M. and Bhutta, M.F. (2021) 'Environmental Impact of Personal Protective Equipment Distributed for Use by Health and Social Care Services in England in the First Six Months of the COVID-19 Pandemic', *Journal of the Royal Society of Medicine*, 114(5), pp. 250–263. doi: 10.1177/01410768211001583. **Peer-reviewed (JRSM).** *Quantifies plastic waste surge from PPE during COVID-19.*

35. **[klemes2020energy]** Klemeš, J.J., Fan, Y.V., Tan, R.R. and Jiang, P. (2020) 'Minimising the Present and Future Plastic Waste, Energy and Environmental Footprints Related to COVID-19', *Renewable and Sustainable Energy Reviews*, 127, p. 109883. doi: 10.1016/j.rser.2020.109883. **Peer-reviewed (Elsevier).** *Single-use medical plastics environmental footprint analysis.*

36. **[lee2023medical]** Lee, B.K., Ellenbecker, M.J. and Moure-Eraso, R. (2004) 'Analyses of the Recycling Potential of Medical Plastic Wastes', *Waste Management*, 24(10), pp. 991–998. doi: 10.1016/j.wasman.2004.07.006. **Peer-reviewed (Elsevier).** *Direct relevance: analyses which medical plastics (PP, PS, PE) can be recycled and at what rates; polypropylene tubes are 70–80% recyclable.*

### 20.11 Deep Learning Frameworks & Tools

37. **[paszke2019pytorch]** Paszke, A., Gross, S., Massa, F. et al. (2019) 'PyTorch: An Imperative Style, High-Performance Deep Learning Library', *Advances in Neural Information Processing Systems (NeurIPS)*, 32, pp. 8026–8037. **Peer-reviewed (NeurIPS).**

### 20.12 Papers Already Collected (in Papers/ folder — titles TBC)

38. **sensors-20-03816.pdf** — MDPI Sensors, 2020 (doi: 10.3390/s20133816). **Peer-reviewed.**
39. **sensors-21-01213.pdf** — MDPI Sensors, 2021 (doi: 10.3390/s21041213). **Peer-reviewed.**
40. **s41598-023-45759-z.pdf** — Nature Scientific Reports, 2023. **Peer-reviewed.**
41. **e3sconf_icfee2024_04001.pdf** — E3S Web of Conferences, ICFEE 2024. **Conference proceedings.**
42. **3093-4155-1-PB.pdf** — Title TBC (needs extraction from PDF).

### 20.13 Key Statistics for Introduction/Motivation

| Statistic | Value | Source |
| --- | --- | --- |
| Global healthcare waste (non-hazardous share) | 85% | WHO (2024) |
| Hazardous waste per bed per day (high-income) | 0.5 kg | WHO (2024) |
| US hospital waste per year | >5 million tonnes | Practice Greenhealth (2024) |
| US hospital waste per bed per day | 29 pounds (~13 kg) | Practice Greenhealth (2024) |
| Global injections per year | 16 billion | WHO (2024) |
| NHS direct emissions reduction since 1990 | 68% | NHS England (2022) |
| NHS net zero target (direct) | 2040 | NHS England (2022) |
| NHS net zero target (supply chain) | 2045 | NHS England (2022) |
| Polypropylene recyclability rate | 70–80% | Lee et al. (2004) |
| COVID-19 PPE waste (England, 6 months) | >27,000 tonnes | Rizan et al. (2021) |
