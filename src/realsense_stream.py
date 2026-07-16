import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import argparse
from pathlib import Path

_ULTRALYTICS_CONFIG_DIR = Path(__file__).resolve().parent.parent / ".ultralytics"
_ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR = Path(__file__).resolve().parent.parent / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

from ultralytics import YOLO  # noqa: E402

# --- Stream configuration (USB 3 — matches capture_dataset.py) ---
COLOR_WIDTH  = 1280
COLOR_HEIGHT = 720
COLOR_FPS    = 30
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30

# Scene depth calibration — actual capture setup
SURFACE_DISTANCE_MM   = 480   # camera raised 30 mm → surface now at ~480 mm
TUBE_MAX_THICKNESS_MM = 30
DEPTH_NEAR_MARGIN_MM  = 20
DEPTH_FAR_MARGIN_MM   = 40
DEPTH_MIN_MM = SURFACE_DISTANCE_MM - TUBE_MAX_THICKNESS_MM - DEPTH_NEAR_MARGIN_MM  # 430 mm
DEPTH_MAX_MM = SURFACE_DISTANCE_MM + DEPTH_FAR_MARGIN_MM                           # 520 mm

# Depth ROI crop — aligns both panels to the valid depth FOV
DEPTH_FOV_PAD_PX    = 2
MIN_VALID_DEPTH_PIX = 1000
MIN_VALID_COL_RATIO = 0.25
MIN_VALID_ROW_RATIO = 0.10

# Class colours (BGR) keyed by name from data.yaml
# 0: Other → yellow   1: Push-on → blue   2: Screwcap → green   3: Universal → red
CLASS_COLORS = {
    "Other":     (  0, 220, 220),   # yellow
    "Push-on":   (220,   0,   0),   # blue
    "Screwcap":  (  0, 200,   0),   # green
    "Universal": (  0,   0, 220),   # red
}
DEFAULT_MASK_COLOR = (180, 60, 255)  # purple fallback for unknown classes

# Mask overlay on depth panel
MASK_ALPHA = 0.55

# HUD bar
HUD_H      = 44
HUD_MARGIN = 32

# Recording
REC_SAVE_INTERVAL_S = 0.5   # min seconds between auto-saved frames while recording

# Grid layout
GRID_PANEL_W  = 800   # each quadrant width in pixels
GRID_GAP      = 6     # gap in pixels between panels (dark separator)

RECONNECT_DELAY  = 3
MAX_RETRIES      = 10
DEFAULT_WEIGHTS  = "yolo26n.pt"
FALLBACK_WEIGHTS = "runs/segment/runs/2026-07-12_22-48-54/YOLOv8-seg/weights/best.pt"
DEFAULT_SAVE_DIR  = Path("runs/captures")
SNAPSHOT_SUBDIR   = "snapshots"
MODEL_IMG_SIZE   = 640
MODEL_CONF       = 0.35


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense live overlay stream with YOLO inference.")
    parser.add_argument("--weights", default="", help="Path to a local YOLO .pt file")
    return parser.parse_args()


def build_config():
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    return config


# ── Sensor setup ──────────────────────────────────────────────────────────────

def _set_opt(sensor, option, value: float, label: str):
    if not sensor.supports(option):
        return
    try:
        rng = sensor.get_option_range(option)
        sensor.set_option(option, min(max(value, rng.min), rng.max))
    except RuntimeError as e:
        print(f"[warn] Could not set {label}: {e}")


def configure_color_sensor(profile):
    """Enable auto-exposure + auto-white-balance to match dataset capture settings."""
    device = profile.get_device()
    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name).lower()
        except RuntimeError:
            continue
        if "rgb" in name or "color" in name:
            _set_opt(sensor, rs.option.enable_auto_exposure,      1.0, "enable_auto_exposure")
            _set_opt(sensor, rs.option.enable_auto_white_balance,  1.0, "enable_auto_white_balance")
            print("[info] Color sensor: auto-exposure + auto-white-balance enabled")
            return
    print("[warn] Color sensor not found; skipping exposure config")


# ── Depth colourmap ─────────────────────────────────────────────────────────

def depth_to_preview(depth_raw: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """COLORMAP_TURBO clipped to the given depth range in mm."""
    valid   = depth_raw > 0
    clipped = np.clip(depth_raw, d_min, d_max).astype(np.float32)
    img8    = ((clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    img8[~valid] = 0
    colour  = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def compute_depth_valid_roi(depth_mm: np.ndarray, pad_px: int = DEPTH_FOV_PAD_PX):
    """Return (y0, y1, x0, x1) bounding box of the valid depth region, or None."""
    valid = depth_mm > 0
    if valid.sum() < MIN_VALID_DEPTH_PIX:
        return None
    h, w = depth_mm.shape[:2]
    min_col = max(1, int(h * MIN_VALID_COL_RATIO))
    min_row = max(1, int(w * MIN_VALID_ROW_RATIO))
    strong_cols = np.where(valid.sum(axis=0) >= min_col)[0]
    strong_rows = np.where(valid.sum(axis=1) >= min_row)[0]
    if not strong_cols.size or not strong_rows.size:
        return None
    x0 = max(0,     int(strong_cols.min()) - pad_px)
    x1 = min(w,     int(strong_cols.max()) + 1 + pad_px)
    y0 = max(0,     int(strong_rows.min()) - pad_px)
    y1 = min(h,     int(strong_rows.max()) + 1 + pad_px)
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None
    return (y0, y1, x0, x1)


# ── Polygon mask helper ───────────────────────────────────────────────────────────

def get_mask_binary(seg_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """Rasterize masks.xy polygon (original-image coords) — no resize artefacts."""
    if len(seg_xy) == 0:
        return np.zeros((h, w), dtype=bool)
    pts = seg_xy.astype(np.int32).reshape(-1, 1, 2)
    bm  = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(bm, [pts], (1,))
    return bm.astype(bool)


# ── Overlay drawing with class colours ───────────────────────────────────────

def draw_overlay(color_image: np.ndarray, result) -> np.ndarray:
    """Draw YOLO segmentation results using class-specific colours (no default palette)."""
    out   = color_image.copy()
    if result.masks is None or result.boxes is None:
        return out
    h, w    = out.shape[:2]
    names   = result.names
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs   = result.boxes.conf.cpu().numpy()
    boxes   = result.boxes.xyxy.cpu().numpy().astype(int)
    for seg_xy, cls_id, conf, box in zip(result.masks.xy, cls_ids, confs, boxes):
        name   = names.get(cls_id, "")
        color  = CLASS_COLORS.get(name, DEFAULT_MASK_COLOR)
        binary = get_mask_binary(seg_xy, h, w)
        fill   = out.copy()
        fill[binary] = color
        out = cv2.addWeighted(out, 1.0 - MASK_ALPHA, fill, MASK_ALPHA, 0)
        cm  = (binary.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        tc = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(out, lbl, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, tc, 1, cv2.LINE_AA)
    return out


# ── Mask overlay on depth ──────────────────────────────────────────────────────

def overlay_masks_on_depth(depth_display: np.ndarray, result) -> np.ndarray:
    """Blend YOLO segmentation masks + class labels over the depth colourmap."""
    out = depth_display.copy()
    if result.masks is None or result.boxes is None:
        return out
    dh, dw  = out.shape[:2]
    names   = result.names
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs   = result.boxes.conf.cpu().numpy()
    boxes   = result.boxes.xyxy.cpu().numpy().astype(int)
    for seg_xy, cls_id, conf, box in zip(result.masks.xy, cls_ids, confs, boxes):
        name    = names.get(cls_id, "")
        color   = CLASS_COLORS.get(name, DEFAULT_MASK_COLOR)
        binary  = get_mask_binary(seg_xy, dh, dw)
        overlay = out.copy()
        overlay[binary] = color
        out = cv2.addWeighted(out, 1.0 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
        cm  = (binary.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
        # Class label on depth panel
        x1, y1, x2, y2 = box
        lbl = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        tc = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(out, lbl, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, tc, 1, cv2.LINE_AA)
    return out


# ── HUD ───────────────────────────────────────────────────────────────────────

def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a readable title label with a semi-transparent dark background."""
    out  = img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    sc   = 0.6
    (tw, th), bl = cv2.getTextSize(text, font, sc, 1)
    pad  = 7
    bg   = out.copy()
    cv2.rectangle(bg, (0, 0), (tw + pad * 2, th + bl + pad * 2), (18, 18, 18), -1)
    out  = cv2.addWeighted(out, 0.35, bg, 0.65, 0)
    cv2.putText(out, text, (pad, th + pad), font, sc, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def build_grid(tl: np.ndarray, tr: np.ndarray,
               bl: np.ndarray, br: np.ndarray,
               panel_w: int, gap: int = GRID_GAP) -> np.ndarray:
    """Arrange four panels into a 2×2 grid with pixel gaps between them."""
    def _scale(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (panel_w, int(h * panel_w / w)))
    tl, tr, bl, br = _scale(tl), _scale(tr), _scale(bl), _scale(br)
    ph = tl.shape[0]
    sep_col = np.full((ph,         gap, 3), 25, dtype=np.uint8)
    sep_row = np.full((gap, panel_w * 2 + gap, 3), 25, dtype=np.uint8)
    top    = np.hstack((tl, sep_col, tr))
    bottom = np.hstack((bl, sep_col, br))
    return np.vstack((top, sep_row, bottom))


def fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_hud(canvas: np.ndarray, recording: bool, rec_elapsed: float,
             snapshot_count: int, rec_count: int, elapsed_s: float,
             fps: float = 0.0, model_name: str = "") -> np.ndarray:
    h, w = canvas.shape[:2]
    bar  = np.zeros((HUD_H, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    if recording:
        state_text  = f"[REC  {fmt_hms(rec_elapsed)}]"
        state_color = (60, 60, 220)
    else:
        state_text  = "[LIVE]"
        state_color = (60, 220, 60)
    fps_text = f"{fps:.1f} FPS" if fps > 0 else ""
    model_text = f"  {model_name}" if model_name else ""
    left  = (f"{state_text}  {fmt_hms(elapsed_s)}  {fps_text}{model_text}    "
             f"Snapshots: {snapshot_count}   Recording: {rec_count} frames")
    right = "Q = Quit   Space = Snap   R = Record   M = Switch Model"
    cv2.putText(bar, left,  (HUD_MARGIN, 32), font, 0.55, state_color, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(right, font, 0.5, 1)
    cv2.putText(bar, right, (w - tw - HUD_MARGIN, 32), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack((canvas, bar))


# ── Main stream loop ─────────────────────────────────────────────────────────

def depth_to_heatmap(depth_raw: np.ndarray) -> np.ndarray:
    """TURBO colourmap auto-ranged to valid depth values per frame (for heatmap panel)."""
    valid = depth_raw > 0
    if not valid.any():
        return np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
    d_min = float(depth_raw[valid].min())
    d_max = float(depth_raw[valid].max())
    if d_max == d_min:
        return np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
    img8 = np.zeros_like(depth_raw, dtype=np.uint8)
    img8[valid] = ((depth_raw[valid].astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    colour = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def stream_loop(pipeline, align, model, save_dir: Path, start_ts: float,
               model_paths: list | None = None, model_index: int = 0):
    snap_dir = save_dir / SNAPSHOT_SUBDIR
    snap_dir.mkdir(parents=True, exist_ok=True)
    snapshot_index = 1
    recording      = False
    rec_start_ts   = 0.0
    rec_count      = 0
    last_rec_save  = 0.0
    rec_dir: Path | None  = None
    depth_roi:   tuple | None = None
    depth_range: tuple | None = None   # (d_min_mm, d_max_mm) auto-estimated from scene
    frame_times: list[float] = []     # rolling window for FPS calculation
    FPS_WINDOW   = 30                 # average over last N frames

    WIN = "MedTube D415  |  TL: RGB   TR: Depth   BL: Masks   BR: Depth+Masks"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1470, 740)
    cv2.moveWindow(WIN, 0, 25)

    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=15000)
        except RuntimeError as e:
            print(f"[warn] Frame error: {e} — reconnecting...")
            return False

        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_mm    = np.asanyarray(depth_frame.get_data())

        # Flip 180° (camera is mounted upside-down)
        color_image = cv2.flip(color_image, -1)
        depth_mm    = cv2.flip(depth_mm,    -1)

        # Lock depth ROI and auto-estimate depth range on first stable frame
        if depth_roi is None:
            roi = compute_depth_valid_roi(depth_mm)
            if roi is not None:
                depth_roi = roi
                y0, y1, x0, x1 = roi
                roi_depth  = depth_mm[y0:y1, x0:x1]
                valid_d    = roi_depth[(roi_depth > 0) & (roi_depth < 3000)].astype(np.float32)
                if valid_d.size < 100:
                    valid_d = roi_depth[roi_depth > 0].astype(np.float32)
                d_lo       = float(np.percentile(valid_d, 5))
                d_hi       = float(np.percentile(valid_d, 75))    # 75th cuts out far walls
                depth_range = (max(0.0, d_lo - 20.0), d_hi + 20.0)
                print(f"[info] Depth ROI locked: x={x0}:{x1}, y={y0}:{y1}")
                print(f"[info] Depth range auto-estimated: {depth_range[0]:.0f}–{depth_range[1]:.0f} mm")

        results      = model.predict(color_image, imgsz=MODEL_IMG_SIZE, conf=MODEL_CONF, verbose=False)
        img_overlay  = draw_overlay(color_image, results[0])
        depth_heat    = depth_to_heatmap(depth_mm)
        depth_preview = (depth_to_preview(depth_mm, *depth_range) if depth_range
                         else depth_to_heatmap(depth_mm))
        depth_masked  = overlay_masks_on_depth(depth_preview.copy(), results[0])

        # Crop all frames to depth ROI if locked (only if ROI is reasonably sized)
        if depth_roi is not None:
            y0, y1, x0, x1 = depth_roi
            roi_h = y1 - y0
            # Only crop if ROI covers at least 40% of frame height (avoid thin strips)
            if roi_h >= color_image.shape[0] * 0.4:
                c_stream  = color_image[y0:y1, x0:x1]
                c_overlay = img_overlay[y0:y1, x0:x1]
                c_depth   = depth_masked[y0:y1, x0:x1]
                c_heat    = depth_heat[y0:y1, x0:x1]
            else:
                c_stream  = color_image
                c_overlay = img_overlay
                c_depth   = depth_masked
                c_heat    = depth_heat
        else:
            c_stream  = color_image
            c_overlay = img_overlay
            c_depth   = depth_masked
            c_heat    = depth_heat

        now = time.time()

        # Auto-save all four views during recording
        if recording and (now - last_rec_save) >= REC_SAVE_INTERVAL_S:
            assert rec_dir is not None
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(rec_dir / f"stream_{ts}_{rec_count:04d}.png"),  c_stream)
            cv2.imwrite(str(rec_dir / f"overlay_{ts}_{rec_count:04d}.png"), c_overlay)
            cv2.imwrite(str(rec_dir / f"depth_{ts}_{rec_count:04d}.png"),   c_depth)
            cv2.imwrite(str(rec_dir / f"heat_{ts}_{rec_count:04d}.png"),    c_heat)
            rec_count    += 1
            last_rec_save = now

        rec_elapsed = (now - rec_start_ts) if recording else 0.0

        # FPS calculation (rolling window)
        frame_times.append(now)
        if len(frame_times) > FPS_WINDOW:
            frame_times = frame_times[-FPS_WINDOW:]
        fps = len(frame_times) / max(frame_times[-1] - frame_times[0], 1e-6) if len(frame_times) > 1 else 0.0

        # Model display name — show architecture (e.g. "YOLOv8m-seg") not just filename
        _ckpt = str(getattr(model, 'ckpt_path', '') or '')
        _yaml = getattr(model.model, 'yaml', {}).get('yaml_file', '')
        if _yaml:
            model_name = Path(_yaml).stem  # e.g. "yolov8m-seg", "yolo26n-seg"
        else:
            model_name = Path(_ckpt).stem or 'unknown'

        # Build 2×2 grid with panel labels; HUD spans full width at bottom
        grid = build_grid(
            label_panel(c_stream,  "RGB Stream"),
            label_panel(c_heat,    "Depth Heatmap"),
            label_panel(c_overlay, f"Stream + Masks [{model_name}]"),
            label_panel(c_depth,   f"Depth + Masks [{model_name}]"),
            GRID_PANEL_W,
        )
        grid = draw_hud(grid, recording, rec_elapsed,
                        snapshot_index - 1, rec_count, now - start_ts,
                        fps=fps, model_name=model_name)

        cv2.imshow(WIN, grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if recording:
                print(f"[info] Recording stopped — {rec_count} frames in {rec_dir}")
            return True

        elif key == ord(" "):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(snap_dir / f"stream_{ts}_{snapshot_index:03d}.png"),  c_stream)
            cv2.imwrite(str(snap_dir / f"overlay_{ts}_{snapshot_index:03d}.png"), c_overlay)
            cv2.imwrite(str(snap_dir / f"depth_{ts}_{snapshot_index:03d}.png"),   c_depth)
            cv2.imwrite(str(snap_dir / f"heat_{ts}_{snapshot_index:03d}.png"),    c_heat)
            print(f"[info] Snapshot {snapshot_index} → {snap_dir}")
            snapshot_index += 1

        elif key == ord("r"):
            if not recording:
                ts      = time.strftime("%Y%m%d_%H%M%S")
                rec_dir = save_dir / f"rec_{ts}"
                rec_dir.mkdir(parents=True, exist_ok=True)
                recording     = True
                rec_start_ts  = time.time()
                rec_count     = 0
                last_rec_save = 0.0
                print(f"[info] Recording started → {rec_dir}")
            else:
                recording = False
                print(f"[info] Recording stopped — {rec_count} frames → {rec_dir}")

        elif key == ord("m") and model_paths and len(model_paths) > 1:
            model_index = (model_index + 1) % len(model_paths)
            new_path = model_paths[model_index]
            print(f"[info] Switching model → {new_path.name}")
            model = YOLO(str(new_path))
            frame_times.clear()  # reset FPS counter for new model
            print(f"[info] Loaded {new_path.name}")


def main():
    args    = parse_args()
    align   = rs.align(rs.stream.color)
    retries = 0

    weights_arg = args.weights.strip()
    if weights_arg:
        weights_path = Path(weights_arg)
    else:
        weights_path = Path(os.getenv("MEDTUBE_WEIGHTS", DEFAULT_WEIGHTS))

    if not weights_path.exists() and Path(FALLBACK_WEIGHTS).exists():
        weights_path = Path(FALLBACK_WEIGHTS)

    if not weights_path.exists():
        print(f"[error] Weights not found: {weights_path}")
        print("  Put weights.pt in the repo root or set MEDTUBE_WEIGHTS to a valid local .pt file.")
        return

    model    = YOLO(str(weights_path))
    save_dir = DEFAULT_SAVE_DIR
    print(f"Loading weights from: {weights_path}")

    # Discover all available .pt model files for live switching (M key)
    project_root = Path(__file__).resolve().parent.parent
    model_paths = sorted(
        p for p in project_root.glob("*.pt")
        if p.stem not in ("yolov8m-seg", "yolo11m-seg")  # skip pretrained COCO bases
    )
    # Ensure the active model is in the list and find its index
    if weights_path.resolve() not in [p.resolve() for p in model_paths]:
        model_paths.insert(0, weights_path.resolve())
    model_index = next(
        (i for i, p in enumerate(model_paths) if p.resolve() == weights_path.resolve()), 0
    )
    print(f"[info] Available models for switching (M key): {[p.name for p in model_paths]}")

    while retries < MAX_RETRIES:
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(build_config())
        except RuntimeError as e:
            print(f"[error] Could not start pipeline: {e}")
            print(f"  Retrying in {RECONNECT_DELAY}s... ({retries + 1}/{MAX_RETRIES})")
            time.sleep(RECONNECT_DELAY)
            retries += 1
            continue

        retries  = 0
        start_ts = time.time()
        done     = False
        try:
            configure_color_sensor(profile)
            # Warmup: let auto-exposure settle before streaming
            print("[info] Waiting 4 s for auto-exposure to settle...")
            time.sleep(4.0)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            print(f"[info] Streaming — depth scale: {depth_scale:.6f} m/unit")
            print("[info] Depth colourmap: TURBO  auto-estimated from scene on ROI lock")
            done = stream_loop(pipeline, align, model, save_dir, start_ts,
                               model_paths=model_paths, model_index=model_index)
        finally:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

        if done:
            break

        print(f"  Reconnecting in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)

    cv2.destroyAllWindows()
    print("Stream stopped.")


if __name__ == "__main__":
    main()
