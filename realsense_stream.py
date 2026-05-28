import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- Configuration ---
# USB 2.1 (~480 Mbps) cannot handle two simultaneous 640x480@30fps streams.
# Use 15fps for colour and a smaller depth resolution.
# Switch to USB 3 and raise these to 1280x720 @ 30fps for full quality.
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 15
DEPTH_WIDTH = 480
DEPTH_HEIGHT = 270
DEPTH_FPS = 15

RECONNECT_DELAY = 3   # seconds to wait before retrying after a disconnect
MAX_RETRIES = 10


def build_config():
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    return config


def stream_loop(pipeline, align, colorizer):
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=15000)
        except RuntimeError as e:
            print(f"[warn] Frame error: {e} — reconnecting...")
            return False  # signal reconnect needed

        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Resize depth colourmap to match colour frame height before stacking
        depth_resized = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
        combined = np.hstack((color_image, depth_resized))

        cv2.imshow("RealSense D415 — Colour | Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return True  # signal clean exit


def main():
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    retries = 0

    print("Press Q in the window to quit.")

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

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"Streaming — depth scale: {depth_scale:.4f} m/unit")
        retries = 0  # reset on successful connect

        try:
            done = stream_loop(pipeline, align, colorizer)
        finally:
            pipeline.stop()

        if done:
            break

        print(f"  Reconnecting in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)

    cv2.destroyAllWindows()
    print("Stream stopped.")


if __name__ == "__main__":
    main()
