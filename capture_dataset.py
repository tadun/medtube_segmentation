"""
Medical tube dataset collector — RealSense D415
================================================
Controls
--------
    Space     Save a single frame (recommended for one-by-one capture)
    R         Toggle recording on/off
    S         Save a single frame (alias for Space)
    Q         Quit and save session summary

Output structure
----------------
  dataset/
        tube_<number>/
            rgb/              colour PNGs  (tube_<number>_rgb_001.png …)
            depth/            colourised depth PNGs  (tube_<number>_depth_001.png …)
            tube_<number>.csv timestamp, frame index per saved frame

Run with:
  sudo rs_env/bin/python capture_dataset.py
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

# --- Configuration ---
COLOR_WIDTH  = 1280
COLOR_HEIGHT = 720
COLOR_FPS    = 30
DEPTH_WIDTH  = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS    = 30

# RGB exposure tuning (D415): default to auto exposure to match earlier datasets.
# Set COLOR_ENABLE_AUTO_EXPOSURE=False to use manual exposure/gain values.
COLOR_ENABLE_AUTO_EXPOSURE = True
COLOR_EXPOSURE_US = 220.0
COLOR_GAIN = 64.0
COLOR_ENABLE_AUTO_WHITE_BALANCE = True

RECONNECT_DELAY  = 3
MAX_RETRIES      = 10

# Minimum seconds between auto-saves during continuous recording.
# Increase this if frames are too similar (e.g. 1.0 = 1 frame/s).
SAVE_INTERVAL_S  = 0.5

# Scene priors (mm): table/surface is ~480-510 mm away; thickest tube is ~40 mm.
# We use these to generate depth colour previews.
SURFACE_DISTANCE_MM = 495
TUBE_MAX_THICKNESS_MM = 40
DEPTH_NEAR_MARGIN_MM = 20
DEPTH_FAR_MARGIN_MM = 40
DEPTH_MIN_MM = SURFACE_DISTANCE_MM - TUBE_MAX_THICKNESS_MM - DEPTH_NEAR_MARGIN_MM
DEPTH_MAX_MM = SURFACE_DISTANCE_MM + DEPTH_FAR_MARGIN_MM

# Crop both RGB and depth to the depth-visible area (valid depth > 0).
CROP_TO_DEPTH_FOV = True
DEPTH_FOV_PAD_PX = 2
MIN_VALID_DEPTH_PIXELS = 1000
MIN_VALID_COL_RATIO = 0.10
MIN_VALID_ROW_RATIO = 0.10

# Seconds to count down after pressing Space before recording starts.
# Use this to pull your hands clear of the frame.
RECORD_DELAY_S = 3

DATA_OWNER_UID = int(os.environ["SUDO_UID"]) if "SUDO_UID" in os.environ else None
DATA_OWNER_GID = int(os.environ["SUDO_GID"]) if "SUDO_GID" in os.environ else None


def ensure_user_ownership(path: Path):
    """If run via sudo, reassign files/dirs to the invoking user for easy editing."""
    if DATA_OWNER_UID is None or DATA_OWNER_GID is None:
        return
    try:
        os.chown(path, DATA_OWNER_UID, DATA_OWNER_GID)
    except OSError:
        pass



def create_session_dirs(tube_number: str):
    session_name = f"tube_{tube_number}"
    base = Path("dataset") / session_name
    (base / "rgb").mkdir(parents=True, exist_ok=True)
    (base / "depth").mkdir(parents=True, exist_ok=True)
    ensure_user_ownership(base)
    ensure_user_ownership(base / "rgb")
    ensure_user_ownership(base / "depth")
    return base


def open_csv(base: Path, tube_number: str, append: bool):
    csv_path = base / f"tube_{tube_number}.csv"
    mode = "a" if append else "w"
    f = open(csv_path, mode, newline="")
    ensure_user_ownership(csv_path)
    writer = csv.writer(f)
    if (not append) or csv_path.stat().st_size == 0:
        writer.writerow(["frame_index", "timestamp", "rgb_filename", "depth_filename"])
    return f, writer


def next_frame_index(base: Path, tube_number: str) -> int:
    """Return next frame index based on existing RGB filenames."""
    rgb_dir = base / "rgb"
    pat = re.compile(rf"^tube_{re.escape(tube_number)}_rgb_(\d+)$")
    max_idx = 0
    for p in rgb_dir.glob("*.png"):
        m = pat.match(p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


# ── Frame saving ─────────────────────────────────────────────────────────────

def depth_to_preview(depth_raw: np.ndarray) -> np.ndarray:
    """Create a colorised depth preview tuned for the capture setup."""
    valid_mask = depth_raw > 0
    depth_clipped = np.clip(depth_raw, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    depth_8bit = ((depth_clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
    depth_8bit[~valid_mask] = 0
    depth_colour = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
    depth_colour[~valid_mask] = 0
    return depth_colour


def compute_depth_valid_roi(depth_mm: np.ndarray, pad_px: int = 0):
    """Return (y0, y1, x0, x1) ROI for valid depth pixels, or None if unstable."""
    valid = depth_mm > 0
    ys, xs = np.where(valid)
    if ys.size < MIN_VALID_DEPTH_PIXELS:
        return None

    h, w = depth_mm.shape[:2]
    # Suppress sparse edge noise by requiring a minimum fraction of valid pixels
    # along each row/column before accepting it into the crop.
    min_col_valid = max(1, int(h * MIN_VALID_COL_RATIO))
    min_row_valid = max(1, int(w * MIN_VALID_ROW_RATIO))
    col_counts = valid.sum(axis=0)
    row_counts = valid.sum(axis=1)
    strong_cols = np.where(col_counts >= min_col_valid)[0]
    strong_rows = np.where(row_counts >= min_row_valid)[0]

    if strong_cols.size and strong_rows.size:
        x0 = max(0, int(strong_cols.min()) - pad_px)
        x1 = min(w, int(strong_cols.max()) + 1 + pad_px)
        y0 = max(0, int(strong_rows.min()) - pad_px)
        y1 = min(h, int(strong_rows.max()) + 1 + pad_px)
    else:
        y0 = max(0, int(ys.min()) - pad_px)
        y1 = min(h, int(ys.max()) + 1 + pad_px)
        x0 = max(0, int(xs.min()) - pad_px)
        x1 = min(w, int(xs.max()) + 1 + pad_px)

    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None
    return (y0, y1, x0, x1)


def save_frame(base, writer, frame_idx, tube_number, color_image, depth_mm):
    rgb_stem = f"tube_{tube_number}_rgb_{frame_idx:03d}"
    depth_stem = f"tube_{tube_number}_depth_{frame_idx:03d}"
    depth_colour = depth_to_preview(depth_mm)
    rgb_path = base / "rgb" / f"{rgb_stem}.png"
    depth_path = base / "depth" / f"{depth_stem}.png"
    cv2.imwrite(str(rgb_path), color_image)
    cv2.imwrite(str(depth_path), depth_colour)
    ensure_user_ownership(rgb_path)
    ensure_user_ownership(depth_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([frame_idx, timestamp, rgb_stem, depth_stem])


# ── Overlay HUD ──────────────────────────────────────────────────────────────

HUD_H = 44  # height in pixels of the info bar below the frames
HUD_MARGIN = 32  # horizontal padding from each edge


def fmt_hms(seconds: float) -> str:
    s   = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_hud(canvas, recording, counting_down, countdown_remaining,
             saved_count, session_elapsed_s, rec_elapsed_s, label):
    """Append a separate black bar below canvas with status + controls."""
    h, w = canvas.shape[:2]
    bar = np.zeros((HUD_H, w, 3), dtype=np.uint8)

    if counting_down:
        rec_text  = f"[STARTING IN {int(countdown_remaining) + 1}]"
        rec_color = (30, 165, 220)   # amber
    elif recording:
        rec_text  = "[REC]"
        rec_color = (60, 60, 220)    # red
    else:
        rec_text  = "[PAUSED]"
        rec_color = (120, 120, 120)  # grey

    left_text  = (f"{rec_text}  rec {fmt_hms(rec_elapsed_s)}  "
                  f"live {fmt_hms(session_elapsed_s)}    "
                  f"Saved: {saved_count}    [{label}]")
    right_text = "SPACE/S = save frame     R = rec/pause     Q = quit"

    cv2.putText(bar, left_text,
                (HUD_MARGIN, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2, cv2.LINE_AA)

    # Measure right text width so it stays HUD_MARGIN from the right edge
    (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(bar, right_text,
                (w - tw - HUD_MARGIN, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack((canvas, bar))


# ── Pipeline config ───────────────────────────────────────────────────────────

def build_config():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16,  DEPTH_FPS)
    return cfg


def set_sensor_option_safe(sensor, option, value: float, label: str):
    """Set sensor option with range-clamp and graceful fallback if unsupported."""
    if not sensor.supports(option):
        return
    try:
        rng = sensor.get_option_range(option)
        clamped = min(max(value, rng.min), rng.max)
        sensor.set_option(option, clamped)
        if clamped != value:
            print(f"[info] {label} clamped to {clamped:.2f}")
    except RuntimeError as e:
        print(f"[warn] Could not set {label}: {e}")


def configure_color_sensor(profile):
    """Apply RGB exposure settings to reduce overexposure in captured frames."""
    device = profile.get_device()
    color_sensor = None

    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name).lower()
        except RuntimeError:
            continue
        if "rgb" in name or "color" in name:
            color_sensor = sensor
            break

    if color_sensor is None:
        print("[warn] Color sensor not found; skipping exposure tuning")
        return

    set_sensor_option_safe(
        color_sensor,
        rs.option.enable_auto_exposure,
        1.0 if COLOR_ENABLE_AUTO_EXPOSURE else 0.0,
        "enable_auto_exposure",
    )

    if not COLOR_ENABLE_AUTO_EXPOSURE:
        set_sensor_option_safe(color_sensor, rs.option.exposure, COLOR_EXPOSURE_US, "exposure")
        set_sensor_option_safe(color_sensor, rs.option.gain, COLOR_GAIN, "gain")

    set_sensor_option_safe(
        color_sensor,
        rs.option.enable_auto_white_balance,
        1.0 if COLOR_ENABLE_AUTO_WHITE_BALANCE else 0.0,
        "enable_auto_white_balance",
    )


# ── Main stream / collect loop ────────────────────────────────────────────────

def collect_loop(pipeline, align, base, writer, state):
    """
    state dict keys: frame_idx, saved_count, live_count, recording
    Returns True  → clean quit
    Returns False → reconnect needed
    """
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=15000)
        except RuntimeError as e:
            print(f"[warn] Frame error: {e} — reconnecting…")
            return False

        aligned      = align.process(frames)
        color_frame  = aligned.get_color_frame()
        depth_frame  = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        state["live_count"] += 1

        session_elapsed = time.time() - state["session_start_ts"]
        color_image    = np.asanyarray(color_frame.get_data())
        depth_mm = np.asanyarray(depth_frame.get_data())   # uint16, mm units

        if CROP_TO_DEPTH_FOV and state["depth_roi"] is None:
            roi = compute_depth_valid_roi(depth_mm, pad_px=DEPTH_FOV_PAD_PX)
            if roi is not None:
                state["depth_roi"] = roi
                y0, y1, x0, x1 = roi
                print(f"[info] Depth ROI locked: x={x0}:{x1}, y={y0}:{y1}")

        if CROP_TO_DEPTH_FOV and state["depth_roi"] is not None:
            y0, y1, x0, x1 = state["depth_roi"]
            color_cropped = color_image[y0:y1, x0:x1]
            depth_cropped = depth_mm[y0:y1, x0:x1]
        else:
            color_cropped = color_image
            depth_cropped = depth_mm

        depth_display = depth_to_preview(depth_cropped)

        # Handle countdown -> recording transition
        if state["counting_down"] and not state["recording"]:
            remaining = state["countdown_end"] - time.time()
            if remaining <= 0:
                state["counting_down"]  = False
                state["recording"]      = True
                state["rec_start_ts"]   = time.time() - state["rec_elapsed"]
                print("[info] RECORDING")

        # Auto-save when recording, throttled by SAVE_INTERVAL_S
        now = time.time()
        if state["recording"] and (now - state["last_save_ts"]) >= SAVE_INTERVAL_S:
            save_frame(base, writer, state["frame_idx"], state["tube_number"], color_cropped, depth_cropped)
            state["frame_idx"]    += 1
            state["saved_count"]  += 1
            state["last_save_ts"]  = now

        # Build display — normalised depth alongside colour
        canvas = np.hstack((color_cropped, depth_display))
        rec_elapsed = time.time() - state["rec_start_ts"] if state["recording"] else state["rec_elapsed"]
        countdown_remaining = max(0.0, state["countdown_end"] - time.time()) if state["counting_down"] else 0.0
        canvas = draw_hud(canvas, state["recording"], state["counting_down"], countdown_remaining,
                          state["saved_count"], session_elapsed, rec_elapsed, state["label"])

        # Scale to fill screen width
        dh, dw = canvas.shape[:2]
        scale  = state["screen_w"] / dw
        if scale != 1.0:
            canvas = cv2.resize(canvas, (int(dw * scale), int(dh * scale)))

        cv2.imshow("Dataset Collector - D415 | RGB | Depth", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return True
        elif key == ord("r"):
            if not state["recording"] and not state["counting_down"]:
                state["counting_down"] = True
                state["countdown_end"] = time.time() + RECORD_DELAY_S
                print(f"[info] Recording starts in {RECORD_DELAY_S}s...")
            elif state["recording"]:
                state["rec_elapsed"] = time.time() - state["rec_start_ts"]
                state["recording"]     = False
                state["counting_down"] = False
                print("[info] PAUSED")
        elif key == ord(" ") or key == ord("s"):
            save_frame(base, writer, state["frame_idx"], state["tube_number"], color_cropped, depth_cropped)
            state["frame_idx"]   += 1
            state["saved_count"] += 1
            print(f"[info] Saved frame {state['frame_idx'] - 1}  (total: {state['saved_count']})")


# ── Entry point ───────────────────────────────────────────────────────────────

def get_screen_width() -> int:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, timeout=5
        )
        m = re.search(r"Resolution:\s*(\d+)\s*x\s*\d+", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 1600


def main():
    print("Dataset Collector — RealSense D415")
    tube_number = input("Tube number (e.g. 1): ").strip()
    if not tube_number:
        tube_number = "1"

    base = create_session_dirs(tube_number)
    start_idx = next_frame_index(base, tube_number)
    append_mode = start_idx > 1
    csv_file, writer = open_csv(base, tube_number, append=append_mode)
    ensure_user_ownership(base / f"tube_{tube_number}.csv")
    print(f"[info] Session folder: {base}")
    if append_mode:
        print(f"[info] Append mode — next frame index: {start_idx}")
    else:
        print("[info] New dataset — starting at frame index 1")
    print("[info] Controls: SPACE/S=save single  R=record/pause  Q=quit")

    screen_w  = get_screen_width()
    align     = rs.align(rs.stream.color)
    state = {
        "frame_idx":        start_idx,
        "saved_count":      start_idx - 1,
        "live_count":       0,
        "label":            f"tube_{tube_number}",
        "tube_number":      tube_number,
        "recording":        False,
        "counting_down":    False,
        "countdown_end":    0.0,
        "last_save_ts":     0.0,
        "rec_start_ts":     0.0,
        "rec_elapsed":      0.0,
        "depth_roi":        None,
        "session_start_ts": time.time(),
        "screen_w":         screen_w,
    }

    retries = 0
    while retries < MAX_RETRIES:
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(build_config())
            configure_color_sensor(profile)
        except RuntimeError as e:
            print(f"[error] Could not start pipeline: {e}")
            print(f"  Retrying in {RECONNECT_DELAY}s… ({retries + 1}/{MAX_RETRIES})")
            time.sleep(RECONNECT_DELAY)
            retries += 1
            continue

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"[info] Streaming — depth scale: {depth_scale:.6f} m/unit")
        retries = 0

        try:
            done = collect_loop(pipeline, align, base, writer, state)
        finally:
            pipeline.stop()

        if done:
            break

        print(f"[info] Reconnecting in {RECONNECT_DELAY}s…")
        time.sleep(RECONNECT_DELAY)

    csv_file.close()
    cv2.destroyAllWindows()
    print(f"[info] Session complete — {state['saved_count']} frames saved to {base}")


if __name__ == "__main__":
    main()
