"""Generate depth-only and RGB-D training datasets from existing RGB + depth captures.

Converts the raw RealSense captures (RGB + aligned depth) into training-ready
formats for depth ablation experiments.

Outputs:
  - depth_only/: TURBO heatmap images (same YOLO labels, depth visualization only)
  - rgbd/: 4-channel RGBD images (RGB + normalised depth as 4th channel)

Usage:
    rs_env/bin/python src/prepare_depth_dataset.py \
        --source dataset/ \
        --output runs/depth_experiment/

Requires paired rgb_*.png + depth_*.png in each tube_* folder.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def depth_to_turbo(depth_mm: np.ndarray) -> np.ndarray:
    """Convert raw depth (mm) to TURBO heatmap for depth-only training."""
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    vals = depth_mm[valid].astype(np.float32)
    d_min = float(np.percentile(vals, 2))
    d_max = float(np.percentile(vals, 98))
    if d_max <= d_min:
        d_max = d_min + 1.0
    img8 = np.zeros_like(depth_mm, dtype=np.uint8)
    img8[valid] = ((vals - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def make_rgbd(rgb: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """Create 4-channel RGBD image (uint8). Depth normalised to 0–255."""
    valid = depth_mm > 0
    if valid.any():
        vals = depth_mm[valid].astype(np.float32)
        d_min, d_max = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
        if d_max <= d_min:
            d_max = d_min + 1.0
        d_norm = np.zeros_like(depth_mm, dtype=np.uint8)
        d_norm[valid] = ((vals - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(depth_mm, dtype=np.uint8)
    return np.dstack((rgb, d_norm))


def main():
    parser = argparse.ArgumentParser(description="Prepare depth-only and RGB-D datasets.")
    parser.add_argument("--source", default="dataset", help="Source directory with tube_* folders")
    parser.add_argument("--output", default="runs/depth_experiment", help="Output directory")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    depth_dir = out / "depth_only" / "images"
    rgbd_dir  = out / "rgbd" / "images"
    depth_dir.mkdir(parents=True, exist_ok=True)
    rgbd_dir.mkdir(parents=True, exist_ok=True)

    tube_dirs = sorted(src.glob("tube_*"))
    total = 0

    for tube in tube_dirs:
        rgb_files = sorted(tube.glob("rgb_*.png"))
        for rgb_path in rgb_files:
            # Find matching depth file
            depth_path = tube / rgb_path.name.replace("rgb_", "depth_")
            if not depth_path.exists():
                continue

            rgb = cv2.imread(str(rgb_path))
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth_mm is None:
                continue

            stem = f"{tube.name}_{rgb_path.stem}"

            # Depth-only (TURBO heatmap)
            turbo = depth_to_turbo(depth_mm)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), turbo)

            # RGB-D (4-channel — save as 4-channel PNG)
            rgbd = make_rgbd(rgb, depth_mm)
            cv2.imwrite(str(rgbd_dir / f"{stem}.png"), rgbd)

            total += 1

    print(f"\033[36m[info]\033[0m Processed {total} paired frames from {len(tube_dirs)} tubes")
    print(f"  Depth-only: {depth_dir}")
    print(f"  RGB-D:      {rgbd_dir}")
    print()
    print("Next steps:")
    print("  1. Copy existing YOLO labels to depth_only/labels/ and rgbd/labels/")
    print("  2. Create data.yaml pointing to these directories")
    print("  3. Train: YOLO('yolo26n.pt').train(data='depth_only/data.yaml', ...)")


if __name__ == "__main__":
    main()
