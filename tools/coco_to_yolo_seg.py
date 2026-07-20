"""
Convert COCO segmentation JSON → YOLO segmentation label files.

Usage:
    python tools/coco_to_yolo_seg.py
"""

import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    {
        "name": "balanced",
        "src":  PROJECT_ROOT / "balanced",
        "dst":  PROJECT_ROOT / "balanced_yolo",
        "splits": ["train", "valid", "test"],
        # Category id=0 is Roboflow null sentinel — skip it
        # COCO ids 1..7 → YOLO ids 0..6
        "skip_ids": {0},
    },
    {
        "name": "depth",
        "src":  PROJECT_ROOT / "depth_coco",
        "dst":  PROJECT_ROOT / "depth_yolo",
        "splits": ["train", "valid", "test"],
        # Category id=0 is Roboflow null sentinel — skip it
        # COCO ids 1..4 → YOLO ids 0..3
        "skip_ids": {0},
    },
]


def coco_seg_to_yolo(ann_path: Path, img_dst: Path, lbl_dst: Path,
                     src_img_dir: Path, skip_ids: set):
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        coco = json.load(f)

    # Build id → filename map
    id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in skip_ids:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    copied = 0
    for img_id, img_info in id_to_info.items():
        fname = img_info["file_name"]
        w, h  = img_info["width"], img_info["height"]
        stem  = Path(fname).stem

        # Copy image
        src_img = src_img_dir / fname
        if src_img.exists():
            shutil.copy2(src_img, img_dst / fname)
            copied += 1

        # Write label (empty file if no annotations)
        label_lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_id   = ann["category_id"]
            yolo_cls = cat_id - 1          # shift: COCO 1-based → YOLO 0-based

            # Segmentation: keep only list-type polygons (skip RLE strings)
            segs = [s for s in ann.get("segmentation", [])
                    if isinstance(s, list) and len(s) >= 6]
            if not segs:
                continue
            poly = max(segs, key=len)      # longest polygon
            poly = [float(v) for v in poly]   # ensure numeric

            # Normalise coordinates
            pts = []
            for i in range(0, len(poly), 2):
                px = max(0.0, min(1.0, poly[i]     / w))
                py = max(0.0, min(1.0, poly[i + 1] / h))
                pts.extend([f"{px:.6f}", f"{py:.6f}"])

            label_lines.append(f"{yolo_cls} " + " ".join(pts))

        with open(lbl_dst / f"{stem}.txt", "w") as f:
            f.write("\n".join(label_lines))

    return copied, len(id_to_info)


def write_data_yaml(dst: Path, nc: int, names: list[str]):
    abs_path = dst.resolve()
    yaml = (
        f"train: {abs_path}/train/images\n"
        f"val:   {abs_path}/valid/images\n"
        f"test:  {abs_path}/test/images\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    (dst / "data.yaml").write_text(yaml)


def get_names(ann_path: Path, skip_ids: set) -> list[str]:
    with open(ann_path) as f:
        coco = json.load(f)
    cats = sorted(
        (c for c in coco["categories"] if c["id"] not in skip_ids),
        key=lambda c: c["id"]
    )
    return [c["name"] for c in cats]


def main():
    for ds in DATASETS:
        src, dst = ds["src"], ds["dst"]
        print(f"\n{'='*60}")
        print(f"Converting: {ds['name']}")

        # Get class names from train split
        train_ann = src / "train" / "_annotations.coco.json"
        names = get_names(train_ann, ds["skip_ids"])
        nc = len(names)
        print(f"  Classes ({nc}): {names}")

        for split in ds["splits"]:
            ann_path    = src / split / "_annotations.coco.json"
            src_img_dir = src / split
            img_dst     = dst / split / "images"
            lbl_dst     = dst / split / "labels"

            copied, total = coco_seg_to_yolo(
                ann_path, img_dst, lbl_dst, src_img_dir, ds["skip_ids"]
            )
            print(f"  {split:6s}: {total} images, {copied} copied")

        write_data_yaml(dst, nc, names)
        print(f"  data.yaml written → {dst}/data.yaml")


if __name__ == "__main__":
    main()
