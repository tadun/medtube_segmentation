"""
Medical tube dataset collector — RealSense D415
================================================
Controls
--------
  Space     Toggle recording on/off
  S         Save a single frame (works whether recording or paused)
  Q         Quit and save session summary

Output structure
----------------
  dataset/
    <YYYYMMDD_HHMMSS>/
      rgb/              colour PNGs  (frame_00000.png …)
      depth_raw/        16-bit PNGs  (frame_00000.png …)
      depth_colour/     colourised PNGs for visual inspection
      metadata.csv      timestamp, frame index per saved frame

Run with:
  sudo rs_env/bin/python collect_dataset.py
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import csv
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

RECONNECT_DELAY  = 3
MAX_RETRIES      = 10

# Minimum seconds between auto-saves during continuous recording.
# Increase this if frames are too similar (e.g. 1.0 = 1 frame/s).
SAVE_INTERVAL_S  = 0.5

# Depth display range in mm (does NOT affect saved raw data).
# Bright = close, dark = far, black = no IR return.
# D415 hardware minimum reliable range is ~280 mm.
# Widen MAX if objects are further away.
DEPTH_MIN_MM = 150
DEPTH_MAX_MM = 600

# Centre-crop factor applied before saving AND display.
# 1.0 = full frame. 0.6 = use centre 60% of each axis (zoom in ~1.67x).
# Useful when the camera is further away and tubes appear small.
CROP_FACTOR = 1.0

# Seconds to count down after pressing Space before recording starts.
# Use this to pull your hands clear of the frame.
RECORD_DELAY_S = 3


# ── Centre crop ──────────────────────────────────────────────────────────────

def centre_crop(img: np.ndarray, factor: float) -> np.ndarray:
    """Crop the centre `factor` fraction of img and upscale back to original size."""
    if factor >= 1.0:
        return img
    h, w = img.shape[:2]
    ch, cw = int(h * factor), int(w * factor)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cropped = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)



def create_session_dirs(label: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug      = label.strip().lower().replace(" ", "_") if label.strip() else "session"
    session_name = f"{timestamp}_{slug}"
    base = Path("dataset") / session_name
    (base / "rgb").mkdir(parents=True, exist_ok=True)
    (base / "depth_raw").mkdir(parents=True, exist_ok=True)
    return base


def open_csv(base: Path):
    csv_path = base / "metadata.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["frame_index", "timestamp_s", "filename"])
    return f, writer


# ── Frame saving ─────────────────────────────────────────────────────────────

def save_frame(base, writer, frame_idx, color_image, depth_raw):
    stem = f"frame_{frame_idx:05d}"
    cv2.imwrite(str(base / "rgb"       / f"{stem}.png"), color_image)
    cv2.imwrite(str(base / "depth_raw" / f"{stem}.png"), depth_raw)
    writer.writerow([frame_idx, f"{time.time():.6f}", stem])


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
    right_text = "SPACE = rec/pause     S = save frame     Q = quit"

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


# ── Main stream / collect loop ────────────────────────────────────────────────

def collect_loop(pipeline, align, colorizer, base, writer, state):
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
        depth_raw_full = np.asanyarray(depth_frame.get_data())   # uint16, mm units

        # Apply centre crop to both images before saving and display
        color_image = centre_crop(color_image, CROP_FACTOR)
        depth_raw   = centre_crop(depth_raw_full, CROP_FACTOR)

        # Normalise depth into 8-bit grayscale for display.
        # Pixels with depth == 0 are invalid (no IR return); render them black.
        valid_mask    = depth_raw > 0
        depth_clipped = np.clip(depth_raw, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
        depth_8bit    = ((depth_clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
        depth_8bit[~valid_mask] = 0  # invalid pixels -> black
        depth_display_gray = cv2.resize(depth_8bit, (COLOR_WIDTH, COLOR_HEIGHT))
        depth_display = cv2.cvtColor(depth_display_gray, cv2.COLOR_GRAY2BGR)

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
            save_frame(base, writer, state["frame_idx"], color_image, depth_raw)
            state["frame_idx"]    += 1
            state["saved_count"]  += 1
            state["last_save_ts"]  = now

        # Build display — normalised depth alongside colour
        canvas = np.hstack((color_image, depth_display))
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
        elif key == ord(" "):
            if not state["recording"] and not state["counting_down"]:
                state["counting_down"] = True
                state["countdown_end"] = time.time() + RECORD_DELAY_S
                print(f"[info] Recording starts in {RECORD_DELAY_S}s...")
            elif state["recording"]:
                state["rec_elapsed"] = time.time() - state["rec_start_ts"]
                state["recording"]     = False
                state["counting_down"] = False
                print("[info] PAUSED")
        elif key == ord("s"):
            save_frame(base, writer, state["frame_idx"], color_image, depth_raw)
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
    label = input("Session label (e.g. single, pairs, mixed): ").strip()
    if not label:
        label = "session"

    base = create_session_dirs(label)
    csv_file, writer = open_csv(base)
    print(f"[info] Session folder: {base}")
    print("[info] Controls: SPACE=record/pause  S=save single  Q=quit")

    screen_w  = get_screen_width()
    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()  # kept for potential future use, not used in display
    spatial   = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    state = {
        "frame_idx":        0,
        "saved_count":      0,
        "live_count":       0,
        "label":            label,
        "recording":        False,
        "counting_down":    False,
        "countdown_end":    0.0,
        "last_save_ts":     0.0,
        "rec_start_ts":     0.0,
        "rec_elapsed":      0.0,
        "session_start_ts": time.time(),
        "screen_w":         screen_w,
        "spatial":          spatial,
    }

    retries = 0
    while retries < MAX_RETRIES:
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(build_config())
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
            done = collect_loop(pipeline, align, colorizer, base, writer, state)
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
