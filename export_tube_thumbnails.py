"""Export a clean cropped thumbnail for each tube class for Table 1."""
import cv2
import numpy as np
from pathlib import Path

DATASET = Path("dataset")
OUT_DIR  = Path("report/figures")
OUT_DIR.mkdir(exist_ok=True)

# (tube_number, class_name, output_filename)
TARGETS = [
    (2,  "universal", "fig_class_universal.png"),
    (13, "pushon",    "fig_class_pushon.png"),
    (27, "screwcap",  "fig_class_screwcap.png"),
    (21, "other",     "fig_class_other.png"),
]

THUMB_H = 320          # output thumbnail height (px)
THUMB_W = 240          # output thumbnail width  (px)
PAD     = 30           # padding around detected tube (px)
BRIGHT_THRESH = 90     # fallback grayscale threshold (unused when Otsu succeeds)
MIN_AREA = 3000        # minimum contour area to count as tube


def detect_tube_bbox(img: np.ndarray):
    """Return (x,y,w,h) bounding box of the tube using Otsu, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu picks threshold automatically — works for both light and dark tubes
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large   = [c for c in cnts if cv2.contourArea(c) > MIN_AREA]
    if not large:
        return None
    return cv2.boundingRect(np.vstack(large))


def best_frame(tube_dir: Path) -> np.ndarray | None:
    """Return the frame whose detected tube is closest to image centre."""
    rgb_dir = tube_dir / "rgb"
    frames  = sorted(rgb_dir.glob("*.png"))
    if not frames:
        return None

    cx_img, cy_img = 1280 / 2, 720 / 2
    best_score, best_img = float("inf"), None

    for p in frames[4::5]:
        img  = cv2.imread(str(p))
        bbox = detect_tube_bbox(img)
        if bbox is None:
            continue
        x, y, w, h = bbox
        score = abs((x + w / 2) - cx_img) + abs((y + h / 2) - cy_img)
        if score < best_score:
            best_score, best_img = score, img

    return best_img


def crop_tube(img: np.ndarray) -> np.ndarray:
    """Crop to the tube bounding box with padding, then resize."""
    bbox = detect_tube_bbox(img)
    if bbox is None:
        h, w = img.shape[:2]
        crop = img[h//4:3*h//4, w//4:3*w//4]
    else:
        x, y, bw, bh = bbox
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(img.shape[1], x + bw + PAD)
        y2 = min(img.shape[0], y + bh + PAD)
        crop = img[y1:y2, x1:x2]

    return cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_LANCZOS4)


for tube_num, cls, fname in TARGETS:
    tube_dir = DATASET / f"tube_{tube_num}"
    img = best_frame(tube_dir)
    if img is None:
        print(f"[warn] No frames found for tube_{tube_num}")
        continue
    thumb = crop_tube(img)
    out   = OUT_DIR / fname
    cv2.imwrite(str(out), thumb)
    print(f"[ok] {fname}  ({thumb.shape[1]}×{thumb.shape[0]})")
