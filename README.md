# MedTube Segmentation

Dataset collection and segmentation pipeline for medical tubes using an Intel RealSense D415 depth camera.

---

## Hardware

| Component | Details |
|---|---|
| Camera | Intel RealSense D415 |
| Connection | USB 3.2 |
| Capture resolution | 1280 × 720 @ 30 fps |

> **macOS note:** `pyrealsense2` requires elevated privileges to claim the USB interface.
> Always prefix scripts with `sudo rs_env/bin/python`.

---

## Setup

```bash
# Create and activate the virtual environment
python3.12 -m venv rs_env
source rs_env/bin/activate

# Install dependencies
pip install pyrealsense2 opencv-python numpy
```

---

## Scripts

### `collect_dataset.py`

Streams RGB and aligned depth from the D415 and saves paired frames to disk.

```bash
sudo rs_env/bin/python collect_dataset.py
```

**Controls**

| Key | Action |
|---|---|
| `Space` | Toggle continuous recording on / off |
| `S` | Save a single frame |
| `Q` | Quit |

**Output**

```
dataset/
  YYYY-MM-DD_HH-MM-SS/
    rgb/           Colour PNGs  (1280 x 720, BGR)
    depth_raw/     16-bit PNGs  (Z16, millimetres)
    metadata.csv   frame index + Unix timestamp per saved frame
```

The preview shows a normalised 8-bit grayscale depth image (bright = close, dark = far, black = no return).
Saved depth files are the original raw 16-bit values.

---

### `realsense_stream.py`

Minimal live preview for verifying the camera feed.

```bash
sudo rs_env/bin/python realsense_stream.py
```

---

## Project Structure

```
medtube_segmentation/
  collect_dataset.py    Dataset collection
  realsense_stream.py   Live preview
  rs_env/               Python virtual environment  (not committed)
  dataset/              Collected frames            (not committed)
```
