"""Generate overlay previews for labels auto-filled from neighboring frames.

This script targets labels that were empty and later auto-filled using option 3.
It discovers those labels via the backup tree:
  MedTube Segmentation.yolov8/label_backups_before_option3/

For each corresponding current label, it renders polygons on the image and saves
annotated previews to:
  MedTube Segmentation.yolov8/qa_autofilled_previews/
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("MedTube Segmentation.yolov8")
BACKUP_DIR = DATASET_DIR / "label_backups_before_option3"
OUTPUT_DIR = DATASET_DIR / "qa_autofilled_previews"


def find_image_for_label(label_path: Path) -> Path | None:
    """Find image file that corresponds to a YOLO label file."""
    split = label_path.parts[-3]
    images_dir = DATASET_DIR / split / "images"
    stem = label_path.stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def draw_yolo_segmentation(image: np.ndarray, label_text: str) -> np.ndarray:
    """Draw YOLO segmentation polygons and class IDs onto image."""
    h, w = image.shape[:2]
    rendered = image.copy()

    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
        except ValueError:
            continue

        if len(coords) % 2 != 0:
            continue

        points: list[list[int]] = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append([x, y])

        if len(points) < 3:
            continue

        poly = np.array(points, dtype=np.int32)
        cv2.polylines(rendered, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            rendered,
            f"cls {class_id}",
            (poly[0][0], max(20, poly[0][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return rendered


def main() -> None:
    if not BACKUP_DIR.exists():
        raise SystemExit(f"Backup directory not found: {BACKUP_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    backup_labels = sorted(BACKUP_DIR.rglob("*.txt"))
    if not backup_labels:
        raise SystemExit("No backup label files found to process.")

    rendered_count = 0
    missing_count = 0

    for backup_label in backup_labels:
        rel = backup_label.relative_to(BACKUP_DIR)
        current_label = DATASET_DIR / rel

        if not current_label.exists():
            print(f"[warn] Missing current label: {current_label}")
            missing_count += 1
            continue

        image_path = find_image_for_label(current_label)
        if image_path is None:
            print(f"[warn] Missing image for label: {current_label}")
            missing_count += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] Failed to read image: {image_path}")
            missing_count += 1
            continue

        label_text = current_label.read_text(encoding="utf-8")
        preview = draw_yolo_segmentation(image, label_text)

        split = current_label.parts[-3]
        out_dir = OUTPUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{current_label.stem}_overlay.jpg"
        cv2.imwrite(str(out_path), preview)
        rendered_count += 1
        print(f"[ok] {out_path}")

    print(
        f"[done] rendered={rendered_count} missing_or_failed={missing_count} "
        f"output={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
