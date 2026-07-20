"""
Split the flat rgbd/ dataset into train/valid/test (70/15/15).

Produces rgbd_split/ with the structure YOLO expects:
  rgbd_split/
    train/images/   train/labels/
    valid/images/   valid/labels/
    test/images/    test/labels/
    data.yaml

Run once locally, then zip rgbd_split/ and upload to Google Drive.

Usage:
    rs_env/bin/python tools/split_rgbd.py
"""

import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC   = PROJECT_ROOT / "rgbd"
DST   = PROJECT_ROOT / "rgbd_split"

SEED      = 42
RATIOS    = {"train": 0.70, "valid": 0.15, "test": 0.15}

def main():
    imgs = sorted(p.stem for p in (SRC / "images").iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"Total images: {len(imgs)}")

    random.seed(SEED)
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * RATIOS["train"])
    n_valid = int(n * RATIOS["valid"])

    splits = {
        "train": imgs[:n_train],
        "valid": imgs[n_train : n_train + n_valid],
        "test":  imgs[n_train + n_valid :],
    }

    for split, stems in splits.items():
        for sub in ("images", "labels"):
            (DST / split / sub).mkdir(parents=True, exist_ok=True)

        for stem in stems:
            # copy image
            for ext in (".png", ".jpg", ".jpeg"):
                src_img = SRC / "images" / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, DST / split / "images" / src_img.name)
                    break
            # copy label
            src_lbl = SRC / "labels" / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, DST / split / "labels" / src_lbl.name)

        print(f"  {split:6s}: {len(stems)}")

    # Write absolute-path data.yaml
    yaml_content = (
        f"train: {(DST / 'train' / 'images').resolve()}\n"
        f"val:   {(DST / 'valid' / 'images').resolve()}\n"
        f"test:  {(DST / 'test'  / 'images').resolve()}\n\n"
        "nc: 4\n"
        "names: ['Other', 'Push-on', 'Screwcap', 'Universal']\n"
    )
    (DST / "data.yaml").write_text(yaml_content)
    print(f"\ndata.yaml written → {DST}/data.yaml")
    print(f"\nNext: zip rgbd_split/ and upload to Google Drive at MyDrive/2026/rgbd_split.zip")

if __name__ == "__main__":
    main()
