"""
Generate three supplementary figures for the report:
  1. fig_dataset_examples.png  — all 30 tube specimens in a 5×6 grid
  2. fig_model_comparison.png  — grouped bar chart of model metrics
  (Pipeline flowchart is now inline TikZ in main.tex)

Run from the project root:
    rs_env/bin/python tools/generate_report_figures.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import cv2

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "report" / "figures"

# ── Class info ──────────────────────────────────────────────────────────────
CLASSES = ["Other", "Push-on", "Screwcap", "Universal"]
CLASS_COLOURS = {
    "Other":    "#e8c000",   # yellow
    "Push-on":  "#3a7fd5",   # blue
    "Screwcap": "#27ae60",   # green
    "Universal":"#e74c3c",   # red
}

# ── All 30 tubes with their dominant class ──────────────────────────────────
TUBE_CLASSES = {
    "tube_1":  "Screwcap",  "tube_2":  "Universal", "tube_3":  "Universal",
    "tube_4":  "Push-on",   "tube_5":  "Push-on",   "tube_6":  "Push-on",
    "tube_7":  "Push-on",   "tube_8":  "Other",      "tube_9":  "Other",
    "tube_10": "Other",     "tube_11": "Universal",  "tube_12": "Push-on",
    "tube_13": "Push-on",   "tube_14": "Push-on",    "tube_15": "Other",
    "tube_16": "Other",     "tube_17": "Other",      "tube_18": "Universal",
    "tube_19": "Universal", "tube_20": "Screwcap",   "tube_21": "Other",
    "tube_22": "Other",     "tube_23": "Push-on",    "tube_24": "Push-on",
    "tube_25": "Other",     "tube_26": "Screwcap",   "tube_27": "Screwcap",
    "tube_28": "Screwcap",  "tube_29": "Screwcap",   "tube_30": "Screwcap",
}

# One representative tube per class — pick a tube whose frame has a label
CLASS_TUBES = {
    "Other":    "tube_15",
    "Push-on":  "tube_13",
    "Screwcap": "tube_1",
    "Universal":"tube_3",
}

# ── Figure 1: dataset examples ───────────────────────────────────────────────

def load_rgb(tube_id: str, filename: str) -> np.ndarray:
    p = PROJECT / "dataset" / tube_id / "rgb" / filename
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_label_and_frame(tube_id: str, class_idx: int):
    """
    Search training, validation, and test label dirs for a frame from tube_id
    that contains class_idx.  Returns (rgb_filename, polygon_pts) or raises.
    """
    for split in ("train", "valid", "test"):
        label_dir = PROJECT / "MedTube-2.yolov8" / split / "labels"
        for lf in sorted(label_dir.glob(f"{tube_id}_rgb_*")):
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or int(parts[0]) != class_idx:
                        continue
                    coords = list(map(float, parts[1:]))
                    pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    # Derive original frame filename from label stem
                    # e.g. tube_15_rgb_030_png_png.rf.HASH → tube_15_rgb_030.png
                    stem = lf.stem  # tube_15_rgb_030_png_png.rf.HASH
                    frame_num = stem.split("_rgb_")[1].split("_")[0]
                    fname = f"{tube_id}_rgb_{frame_num}.png"
                    return fname, pts
    raise RuntimeError(f"No label found for {tube_id} class {class_idx}")


def crop_around_polygon(img: np.ndarray, pts: list, pad_frac: float = 0.35):
    """Return a crop of img centred on the bounding box of pts, with padding."""
    h, w = img.shape[:2]
    xs = [p[0] * w for p in pts]
    ys = [p[1] * h for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    pad_x = max(bw * pad_frac, 40)
    pad_y = max(bh * pad_frac, 40)
    cx0 = max(0, int(x0 - pad_x))
    cy0 = max(0, int(y0 - pad_y))
    cx1 = min(w, int(x1 + pad_x))
    cy1 = min(h, int(y1 + pad_y))
    # Shift polygon coordinates to cropped frame
    shifted = [((p[0]*w - cx0)/(cx1-cx0), (p[1]*h - cy0)/(cy1-cy0)) for p in pts]
    return img[cy0:cy1, cx0:cx1], shifted


def make_dataset_figure():
    """5 rows × 6 columns grid of all 30 tube specimens."""
    NCOLS, NROWS = 6, 5
    tubes = [f"tube_{i}" for i in range(1, 31)]

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(13, 11))
    fig.subplots_adjust(wspace=0.04, hspace=0.18,
                        left=0.01, right=0.99, top=0.95, bottom=0.04)

    for ax, tube_id in zip(axes.flat, tubes):
        cls_name = TUBE_CLASSES[tube_id]
        cls_idx  = CLASSES.index(cls_name)
        colour   = CLASS_COLOURS[cls_name]

        try:
            fname, pts = find_label_and_frame(tube_id, cls_idx)
            img = load_rgb(tube_id, fname)
            img_crop, pts_crop = crop_around_polygon(img, pts, pad_frac=0.30)
            h, w = img_crop.shape[:2]
            ax.imshow(img_crop)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            px = [p[0] * w for p in pts_crop]
            py = [p[1] * h for p in pts_crop]
            poly = plt.Polygon(list(zip(px, py)), closed=True,
                               facecolor=colour + "44",
                               edgecolor=colour, linewidth=2)
            ax.add_patch(poly)
        except Exception as e:
            ax.set_facecolor("#111111")
            ax.text(0.5, 0.5, "?", ha="center", va="center",
                    transform=ax.transAxes, color="white", fontsize=14)

        # Coloured border matching class
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)

        tube_num = tube_id.split("_")[1]
        ax.set_title(f"{tube_num}  {cls_name}", fontsize=7.5,
                     color=colour, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLOURS[c], label=c) for c in CLASSES]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.005))

    out = OUT_DIR / "fig_dataset_examples.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: model comparison bar chart ────────────────────────────────────

# Data from Table 3 (model comparison) in the report
MODELS_DATA = [
    # (label,          input,  box_map5095, mask_map5095, ms_per_img)
    ("YOLOv8m-seg",   "RGB",   0.968,       0.905,        108.5),
    ("YOLO11n-seg",   "RGB",   0.949,       0.820,         21.7),
    ("YOLO26n-seg",   "RGB",   0.951,       0.820,         23.1),
    ("YOLO11n-RGBD",  "RGBD",  0.988,       0.929,         22.4),
    ("YOLO26n-depth", "Depth", 0.919,       0.804,         22.8),
]

MODALITY_COLOURS = {
    "RGB":   "#3a7fd5",
    "RGBD":  "#27ae60",
    "Depth": "#e8a020",
}


def make_comparison_figure():
    labels   = [m[0] for m in MODELS_DATA]
    box_map  = [m[2] for m in MODELS_DATA]
    mask_map = [m[3] for m in MODELS_DATA]
    modality = [m[1] for m in MODELS_DATA]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars1 = ax.bar(x - width/2, box_map,  width,
                   color=[MODALITY_COLOURS[m] for m in modality],
                   alpha=0.92, zorder=3)
    bars2 = ax.bar(x + width/2, mask_map, width,
                   color=[MODALITY_COLOURS[m] for m in modality],
                   alpha=0.55, hatch='///', edgecolor=[MODALITY_COLOURS[m] for m in modality],
                   linewidth=1.0, zorder=3)

    # Value labels above bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP$_{50{-}95}$", fontsize=11)
    ax.set_ylim(0.75, 1.02)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend placed ABOVE the axes — modality colours + metric pattern key
    modality_patches = [mpatches.Patch(color=c, label=f"{m} input")
                        for m, c in MODALITY_COLOURS.items()]
    solid_patch   = mpatches.Patch(facecolor="grey", alpha=0.92,
                                   label="Box mAP$_{50{-}95}$  (solid)")
    hatched_patch = mpatches.Patch(facecolor="grey", alpha=0.55, hatch='///',
                                   edgecolor="grey", linewidth=1.0,
                                   label="Mask mAP$_{50{-}95}$ (hatched)")
    ax.legend(handles=modality_patches + [solid_patch, hatched_patch],
              fontsize=9, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, 1.22),
              framealpha=0.95, edgecolor="black")

    fig.tight_layout()
    out = OUT_DIR / "fig_model_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_dataset_figure()
    make_comparison_figure()
    print("Done.")
