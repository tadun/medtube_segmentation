"""
Comparison evaluation of all available trained models.
Outputs per-model metrics and appends a summary section to docs/notes.md.

Run from the project root:
    rs_env/bin/python tools/eval_comparison.py
"""

import json
import sys
import tempfile
import textwrap
from datetime import date
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
NOTES   = PROJECT / "docs" / "notes.md"

# ---------------------------------------------------------------------------
# Build absolute-path data yamls (the Roboflow ones use broken relative paths)
# ---------------------------------------------------------------------------

def write_tmp_yaml(content: dict) -> str:
    """Write a dict as yaml to a NamedTemporaryFile and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(content, f)
    f.close()
    return f.name

RGB_DATA = write_tmp_yaml({
    "train": str(PROJECT / "MedTube-2.yolov8" / "train" / "images"),
    "val":   str(PROJECT / "MedTube-2.yolov8" / "valid" / "images"),
    "test":  str(PROJECT / "MedTube-2.yolov8" / "test"  / "images"),
    "nc": 4,
    "names": ["Other", "Push-on", "Screwcap", "Universal"],
})

RGBD_DATA  = str(PROJECT / "runs" / "depth_experiment" / "rgbd"       / "data.yaml")
DEPTH_DATA = str(PROJECT / "runs" / "depth_experiment" / "depth_only" / "data.yaml")

# ---------------------------------------------------------------------------
# Models to evaluate
#   key  : display name
#   value: (weight_path, data_yaml, notes_string)
# ---------------------------------------------------------------------------

MODELS = {
    "YOLOv8m-seg  (100 ep, local)": (
        PROJECT / "runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt",
        RGB_DATA,
        "RGB 3-ch | MedTube-2 test split",
    ),
    "YOLOv9c-seg  (32 ep, Colab)": (
        PROJECT / "YOLOv9c-seg/weights/best.pt",
        RGB_DATA,
        "RGB 3-ch | MedTube-2 test split (early stop)",
    ),
    "yolo26n      (nc=4, RGB)": (
        PROJECT / "weights/yolo26n.pt",
        RGB_DATA,
        "RGB 3-ch | MedTube-2 test split",
    ),
    "YOLO11n-RGBD (100 ep, local)": (
        PROJECT / "runs/2026-07-18_rgbd/YOLO11n-RGBD/weights/best.pt",
        RGBD_DATA,
        "RGBD 4-ch | full RGBD split (train=val=test)",
    ),
    "YOLO11n-RGBD (37 ep, Colab)": (
        PROJECT / "weights/YOLO11n-RGBD/weights/best.pt",
        RGBD_DATA,
        "RGBD 4-ch | full RGBD split (train=val=test, early stop)",
    ),
    "yolo26n-depth (nc=4, depth)": (
        PROJECT / "weights/yolo26n_depth.pt",
        DEPTH_DATA,
        "Depth-only 1-ch | full depth split (train=val=test)",
    ),
    "yolo26n_depth-2 (nc=4, depth, local)": (
        PROJECT / "weights/yolo26n_depth-2.pt",
        DEPTH_DATA,
        "Depth-only 1-ch | full depth split (local training)",
    ),
}

SKIPPED = {
    "yolo26n_balanced (nc=7)": "Incompatible class count (nc=7 vs dataset nc=4); "
                               "likely trained on a different label schema.",
}

# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not found — run with rs_env/bin/python")

results_all = {}

for name, (weight, data, note) in MODELS.items():
    weight = Path(weight)
    if not weight.exists():
        print(f"  [SKIP] {name} — weight not found: {weight}")
        results_all[name] = {"error": "weight not found", "note": note}
        continue

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Weight    : {weight}")
    print(f"  Data      : {data}")
    print(f"{'='*60}")

    try:
        model = YOLO(str(weight))
        metrics = model.val(
            data=data,
            split="test",
            imgsz=640,
            batch=8,
            workers=0,
            device="cpu",
            verbose=False,
            plots=False,
            save_json=False,
        )
        r = {
            "box_P":         round(float(metrics.box.mp),   4),
            "box_R":         round(float(metrics.box.mr),   4),
            "box_mAP50":     round(float(metrics.box.map50),4),
            "box_mAP50-95":  round(float(metrics.box.map),  4),
            "mask_P":        round(float(metrics.seg.mp),   4),
            "mask_R":        round(float(metrics.seg.mr),   4),
            "mask_mAP50":    round(float(metrics.seg.map50),4),
            "mask_mAP50-95": round(float(metrics.seg.map),  4),
            "note": note,
        }
        results_all[name] = r
        print(f"  Box  mAP50-95: {r['box_mAP50-95']}   Mask mAP50-95: {r['mask_mAP50-95']}")
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        results_all[name] = {"error": str(exc), "note": note}

# ---------------------------------------------------------------------------
# Build markdown table
# ---------------------------------------------------------------------------

today = date.today().isoformat()

header = (
    f"\n\n## 22. Model Comparison Eval — {today}\n\n"
    "Eval run via `tools/eval_comparison.py` on the local machine (Apple M1 Max, CPU).\n\n"
    "**Important caveats:**\n"
    "- RGB models evaluated on the held-out **test split** of MedTube-2.yolov8 (450 images, nc=4).\n"
    "- RGBD and depth models evaluated on their respective full datasets "
    "(train=val=test in the data yaml — metrics are therefore **upper-bound estimates** for those models).\n"
    "- `yolo26n_balanced` skipped: nc=7 is incompatible with the nc=4 evaluation set.\n\n"
)

rows = []
rows.append("| Model | Input | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | **Mask mAP50-95** | Notes |")
rows.append("|---|---|---|---|---|---|---|---|---|---|---|")

INPUT_TYPE = {
    "YOLOv8m-seg  (100 ep, local)":  "RGB",
    "YOLOv9c-seg  (32 ep, Colab)":   "RGB",
    "yolo26n      (nc=4, RGB)":       "RGB",
    "YOLO11n-RGBD (100 ep, local)":   "RGBD",
    "YOLO11n-RGBD (37 ep, Colab)":    "RGBD",
    "yolo26n-depth (nc=4, depth)":              "Depth",
    "yolo26n_depth-2 (nc=4, depth, local)":   "Depth",
}

for name, r in results_all.items():
    inp = INPUT_TYPE.get(name, "?")
    if "error" in r:
        rows.append(f"| {name} | {inp} | — | — | — | — | — | — | — | — | {r.get('note','')} — ERROR: {r['error']} |")
    else:
        rows.append(
            f"| {name} | {inp} "
            f"| {r['box_P']} | {r['box_R']} | {r['box_mAP50']} | {r['box_mAP50-95']} "
            f"| {r['mask_P']} | {r['mask_R']} | {r['mask_mAP50']} | **{r['mask_mAP50-95']}** "
            f"| {r['note']} |"
        )

skip_section = "\n\n### Skipped models\n\n"
for name, reason in SKIPPED.items():
    skip_section += f"- **{name}**: {reason}\n"

raw_results = "\n\n<details>\n<summary>Raw JSON results</summary>\n\n```json\n"
raw_results += json.dumps(results_all, indent=2)
raw_results += "\n```\n\n</details>\n"

section = header + "\n".join(rows) + skip_section + raw_results

# ---------------------------------------------------------------------------
# Append to notes.md
# ---------------------------------------------------------------------------

with open(NOTES, "a") as f:
    f.write(section)

print("\n\n" + "="*60)
print("Results appended to docs/notes.md")
print("="*60)
print(section)
