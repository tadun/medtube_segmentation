"""Minimal Kaggle training entrypoint with checkpoint resume support.

Usage on Kaggle Notebook:
  python kaggle_train.py --data /kaggle/input/<dataset>/data.yaml --model yolo11m-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


AUG_PRESETS = {
    "strong": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 180.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 5.0,
        "perspective": 0.0005,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "erasing": 0.4,
    },
    "mild": {
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.2,
        "degrees": 20.0,
        "translate": 0.05,
        "scale": 0.25,
        "shear": 2.0,
        "perspective": 0.0001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.2,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.2,
    },
    "none": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation on Kaggle.")
    parser.add_argument("--data", required=True, help="Path to dataset yaml")
    parser.add_argument("--model", default="yolo11m-seg.pt", help="YOLO weights/model id")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="", help="'', 'cpu', or cuda id like '0'")
    parser.add_argument("--project", default="/kaggle/working/runs")
    parser.add_argument("--name", default="kaggle-seg")
    parser.add_argument("--aug-preset", choices=sorted(AUG_PRESETS), default="mild")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <project>/<name>/weights/last.pt if present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    run_dir = Path(args.project) / args.name
    last_ckpt = run_dir / "weights" / "last.pt"

    if args.resume and last_ckpt.exists():
        print(f"[info] Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
        return

    print(f"[info] Starting new run: model={args.model}")
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        **AUG_PRESETS[args.aug_preset],
    )


if __name__ == "__main__":
    main()
