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


def make_rgbd(rgb: np.ndarray, depth_colour: np.ndarray) -> np.ndarray:
    """Create 4-channel RGBD image. Depth is converted from colourised to grayscale."""
    d_gray = cv2.cvtColor(depth_colour, cv2.COLOR_BGR2GRAY)
    return np.dstack((rgb, d_gray))


def main():
    parser = argparse.ArgumentParser(description="Prepare depth-only and RGB-D datasets.")
    parser.add_argument("--source", default="dataset", help="Source directory with tube_* folders")
    parser.add_argument("--labels", default="", help="Directory with YOLO .txt label files to copy")
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
        # Structure: tube_*/rgb/tube_*_rgb_NNN.png + tube_*/depth/tube_*_depth_NNN.png
        rgb_dir = tube / "rgb"
        depth_src_dir = tube / "depth"
        if not rgb_dir.exists() or not depth_src_dir.exists():
            continue

        rgb_files = sorted(rgb_dir.glob("*.png"))
        for rgb_path in rgb_files:
            # Match rgb → depth by replacing _rgb_ with _depth_ in filename
            depth_name = rgb_path.name.replace("_rgb_", "_depth_")
            depth_path = depth_src_dir / depth_name
            if not depth_path.exists():
                continue

            rgb = cv2.imread(str(rgb_path))
            depth_img = cv2.imread(str(depth_path))
            if rgb is None or depth_img is None:
                continue

            # Resize depth to match RGB if needed
            if depth_img.shape[:2] != rgb.shape[:2]:
                depth_img = cv2.resize(depth_img, (rgb.shape[1], rgb.shape[0]))

            stem = f"{tube.name}_{rgb_path.stem}"

            # Depth-only (use captured depth heatmap directly)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_img)

            # RGB-D (4-channel: RGB + depth grayscale)
            rgbd = make_rgbd(rgb, depth_img)
            cv2.imwrite(str(rgbd_dir / f"{stem}.png"), rgbd)

            total += 1

    # ── Copy labels ───────────────────────────────────────────────────────────
    # Look for YOLO label files that match the generated image names
    label_source = Path(args.labels) if args.labels else None
    depth_label_dir = out / "depth_only" / "labels"
    rgbd_label_dir  = out / "rgbd" / "labels"
    depth_label_dir.mkdir(parents=True, exist_ok=True)
    rgbd_label_dir.mkdir(parents=True, exist_ok=True)

    labels_copied = 0
    if label_source and label_source.exists():
        # Match by image stem → label file
        for img_file in depth_dir.glob("*.png"):
            label_file = label_source / (img_file.stem + ".txt")
            if label_file.exists():
                import shutil
                shutil.copy2(label_file, depth_label_dir / label_file.name)
                shutil.copy2(label_file, rgbd_label_dir / label_file.name)
                labels_copied += 1
        print(f"\033[36m[info]\033[0m Copied {labels_copied} label files")
    else:
        print("\033[33m[warn]\033[0m No --labels dir specified; skipping label copy")

    # ── Create data.yaml files ────────────────────────────────────────────────
    data_yaml_template = """train: {images_dir}
val: {images_dir}
test: {images_dir}

nc: 4
names: ['Other', 'Push-on', 'Screwcap', 'Universal']
"""
    for variant in ("depth_only", "rgbd"):
        yaml_path = out / variant / "data.yaml"
        imgs = str((out / variant / "images").resolve())
        yaml_path.write_text(data_yaml_template.format(images_dir=imgs))
        print(f"\033[36m[info]\033[0m Created {yaml_path}")

    print(f"\n\033[36m[info]\033[0m Processed {total} paired frames from {len(tube_dirs)} tubes")
    print(f"  Depth-only: {depth_dir}")
    print(f"  RGB-D:      {rgbd_dir}")
    print()
    print("To train:")
    print(f"  rs_env/bin/python -c \"from ultralytics import YOLO; YOLO('weights/yolo26n.pt').train(data='{out}/depth_only/data.yaml', epochs=50, imgsz=640)\"")
    print(f"  rs_env/bin/python -c \"from ultralytics import YOLO; YOLO('weights/yolo26n.pt').train(data='{out}/rgbd/data.yaml', epochs=50, imgsz=640)\"")


if __name__ == "__main__":
    main()
