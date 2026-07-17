"""Rebalance MedTube dataset by duplicating underrepresented classes.

Scans all splits (train/valid/test) of the existing YOLO export, identifies
images belonging to minority classes (Screwcap, Universal), and creates
copies with unique filenames ready for Roboflow upload.

After running, upload the contents of the output folder to the Roboflow
project, then generate a new version with a 70/15/15 split.

Usage:
    rs_env/bin/python tools/rebalance_classes.py \
        --dataset MedTube-2.yolov8 \
        --output runs/rebalanced_upload
"""

import argparse
import shutil
import uuid
from pathlib import Path


# Class mapping: id → name
CLASSES = {0: "Other", 1: "Push-on", 2: "Screwcap", 3: "Universal"}


def get_primary_class(label_path: Path) -> int | None:
    """Return the class ID of the first annotation line in a label file."""
    text = label_path.read_text().strip()
    if not text:
        return None
    return int(text.splitlines()[0].split()[0])


def count_per_class(label_files: list[Path]) -> dict[int, list[Path]]:
    """Group label files by their primary class."""
    groups: dict[int, list[Path]] = {i: [] for i in CLASSES}
    for lf in label_files:
        cls = get_primary_class(lf)
        if cls is not None and cls in groups:
            groups[cls].append(lf)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Rebalance dataset by duplicating minority classes.")
    parser.add_argument("--dataset", default="MedTube-2.yolov8",
                        help="Path to YOLO dataset folder")
    parser.add_argument("--output", default="runs/rebalanced_upload",
                        help="Output folder for duplicated images + labels")
    args = parser.parse_args()

    ds = Path(args.dataset)
    out = Path(args.output)
    out_images = out / "images"
    out_labels = out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Collect all label files across all splits
    all_labels: list[Path] = []
    for split in ("train", "valid", "test"):
        label_dir = ds / split / "labels"
        if label_dir.exists():
            all_labels.extend(sorted(label_dir.glob("*.txt")))

    groups = count_per_class(all_labels)

    print("\033[1mCurrent class distribution (all splits):\033[0m")
    for cls_id, files in sorted(groups.items()):
        print(f"  {CLASSES[cls_id]:>10}: {len(files)} images")

    # Target: match the largest class
    target = max(len(files) for files in groups.values())
    print(f"\n\033[1mTarget per class: {target}\033[0m\n")

    total_duplicated = 0

    for cls_id, files in sorted(groups.items()):
        deficit = target - len(files)
        if deficit <= 0:
            print(f"  {CLASSES[cls_id]:>10}: balanced (no action)")
            continue

        print(f"  {CLASSES[cls_id]:>10}: need +{deficit} duplicates")

        # Cycle through existing files to fill the deficit
        for i in range(deficit):
            src_label = files[i % len(files)]
            # Find the corresponding image
            src_image = None
            for split in ("train", "valid", "test"):
                candidate = ds / split / "images" / (src_label.stem + ".png")
                if candidate.exists():
                    src_image = candidate
                    break
                candidate = ds / split / "images" / (src_label.stem + ".jpg")
                if candidate.exists():
                    src_image = candidate
                    break

            if src_image is None:
                continue

            # Create unique filename: original_stem.dup_XXXX.ext
            uid = uuid.uuid4().hex[:8]
            new_stem = f"{src_label.stem}.dup_{uid}"
            new_image = out_images / f"{new_stem}{src_image.suffix}"
            new_label = out_labels / f"{new_stem}.txt"

            # Apply minimal augmentation so Roboflow accepts it (not pixel-identical)
            import cv2
            import numpy as np
            img = cv2.imread(str(src_image))
            # Random brightness shift ±3, random noise σ=1
            shift = np.random.randint(-3, 4)
            noise = np.random.normal(0, 1, img.shape).astype(np.int16)
            aug = np.clip(img.astype(np.int16) + shift + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(new_image), aug)
            shutil.copy2(src_label, new_label)
            total_duplicated += 1

    # Verify final counts
    print(f"\n\033[32m[done]\033[0m Created {total_duplicated} duplicated pairs in {out}")
    print(f"  Images: {out_images}")
    print(f"  Labels: {out_labels}")
    print()
    print("\033[1mNext steps:\033[0m")
    print(f"  1. Upload ALL images from {out_images}/ to Roboflow (medtube-2 project)")
    print(f"  2. Upload ALL labels from {out_labels}/ alongside the images")
    print("  3. Generate a new version with 70/15/15 split")
    print("  4. Roboflow will redistribute all images (originals + duplicates) across splits")


if __name__ == "__main__":
    main()
