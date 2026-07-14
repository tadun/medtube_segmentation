"""View COCO segmentation masks with class overlays.

Usage examples:
    rs_env/bin/python tools/view_masks.py \
      --dataset data_coco \
      --annotations _annotations.coco.json \
      --random \
      --save overlays

    rs_env/bin/python tools/view_masks.py \
      --dataset data_coco \
      --annotations _annotations.coco.json \
      --image-file train/images/example.jpg
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay COCO segmentation masks and class names on images."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Root folder of the downloaded COCO dataset export.",
    )
    parser.add_argument(
        "--annotations",
        default="_annotations.coco.json",
        help="COCO annotations JSON filename (default: _annotations.coco.json).",
    )
    parser.add_argument(
        "--image-file",
        default="",
        help="Specific image file_name to render (for example train/images/a.jpg).",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=0,
        help="Specific COCO image id to render.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Render a random image from the dataset.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Directory to save overlay image. If omitted, opens a window.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Mask fill alpha in [0, 1]. Default: 0.35",
    )
    return parser.parse_args()


def build_indexes(coco: dict) -> tuple[dict[int, dict], dict[int, list[dict]], dict[int, str]]:
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    cat_by_id = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    return images_by_id, anns_by_image, cat_by_id


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rnd = np.random.default_rng(seed=class_id)
    # BGR colors for OpenCV
    b, g, r = rnd.integers(40, 255, size=3).tolist()
    return int(b), int(g), int(r)


def draw_segmentation(
    image: np.ndarray,
    anns: list[dict],
    class_names: dict[int, str],
    alpha: float,
) -> np.ndarray:
    out = image.copy()
    h, w = image.shape[:2]

    for ann in anns:
        cls_id = ann.get("category_id", -1)
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        color = color_for_class(cls_id)

        segments = ann.get("segmentation", [])
        if not isinstance(segments, list):
            # RLE path is skipped in this lightweight viewer.
            continue

        for seg in segments:
            if not isinstance(seg, list) or len(seg) < 6:
                continue

            pts = []
            for i in range(0, len(seg), 2):
                x = int(max(0, min(w - 1, seg[i])))
                y = int(max(0, min(h - 1, seg[i + 1])))
                pts.append([x, y])

            poly = np.array(pts, dtype=np.int32)
            if poly.shape[0] < 3:
                continue

            overlay = out.copy()
            cv2.fillPoly(overlay, [poly], color)
            out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
            cv2.polylines(out, [poly], True, color, 2)

            x0, y0 = poly[0]
            cv2.putText(
                out,
                cls_name,
                (x0, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


def pick_image(images_by_id: dict[int, dict], args: argparse.Namespace) -> dict:
    if args.image_id:
        img = images_by_id.get(args.image_id)
        if not img:
            raise SystemExit(f"[error] image_id {args.image_id} not found.")
        return img

    if args.image_file:
        for img in images_by_id.values():
            if img.get("file_name") == args.image_file:
                return img
        raise SystemExit(f"[error] image_file '{args.image_file}' not found.")

    all_images = list(images_by_id.values())
    if not all_images:
        raise SystemExit("[error] No images in annotations JSON.")

    if args.random:
        return random.choice(all_images)

    return all_images[0]


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    ann_path = dataset_dir / args.annotations

    if not ann_path.exists():
        raise SystemExit(f"[error] Missing annotations file: {ann_path}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images_by_id, anns_by_image, class_names = build_indexes(coco)

    image_meta = pick_image(images_by_id, args)
    image_id = image_meta["id"]
    file_name = image_meta["file_name"]
    image_path = dataset_dir / file_name

    if not image_path.exists():
        raise SystemExit(f"[error] Missing image file: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"[error] Failed to read image: {image_path}")

    anns = anns_by_image.get(image_id, [])
    rendered = draw_segmentation(image, anns, class_names, alpha=args.alpha)

    print(f"[info] image_id={image_id} file_name={file_name} annotations={len(anns)}")

    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"overlay_{Path(file_name).stem}.jpg"
        cv2.imwrite(str(out_path), rendered)
        print(f"[ok] saved overlay: {out_path}")
        return

    window = "COCO Segmentation Viewer"
    cv2.imshow(window, rendered)
    print("[info] Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
