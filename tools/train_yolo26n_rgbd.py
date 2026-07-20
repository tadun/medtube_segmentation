"""
Train yolo26n-seg with 4-channel RGBA input (RGB + Depth).

Same channel-patching approach as the YOLO11n-RGBD run:
  - Load yolo26n.pt (seg base, nc=4)
  - Patch first conv: 3ch → 4ch (depth channel initialised to zero)
  - Train on rgbd/data.yaml (RGBA images, 3000 images, nc=4)

Usage:
    rs_env/bin/python tools/train_yolo26n_rgbd.py
"""

import copy
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML    = str(PROJECT_ROOT / "rgbd" / "data.yaml")
BASE_WEIGHT  = str(PROJECT_ROOT / "weights" / "yolo26n.pt")

# ── Load seg base ────────────────────────────────────────────────────────────
model = YOLO(BASE_WEIGHT)

# ── Patch first conv: 3ch → 4ch ──────────────────────────────────────────────
first_conv = model.model.model[0].conv
old_w = first_conv.weight.data          # shape [out, 3, kH, kW]

new_conv = torch.nn.Conv2d(
    4, old_w.shape[0],
    kernel_size=first_conv.kernel_size,
    stride=first_conv.stride,
    padding=first_conv.padding,
    bias=first_conv.bias is not None,
)
with torch.no_grad():
    new_conv.weight[:, :3] = old_w       # copy RGB weights
    new_conv.weight[:, 3:] = 0.0         # depth channel → zero init
    if first_conv.bias is not None:
        new_conv.bias = copy.deepcopy(first_conv.bias)

model.model.model[0].conv = new_conv
model.model.model[0].conv.in_channels = 4
if hasattr(model.model, "yaml"):
    model.model.yaml["ch"] = 4

print(f"Conv patched: 3ch → 4ch | weight shape: {new_conv.weight.shape}")

# ── Train ────────────────────────────────────────────────────────────────────
model.train(
    data        = DATA_YAML,
    epochs      = 100,
    imgsz       = 640,
    batch       = 8,
    patience    = 50,
    save_period = 10,
    workers     = 0,
    device      = "cpu",
    project     = str(PROJECT_ROOT / "runs" / "yolo26n_rgbd"),
    name        = "yolo26n-RGBD",
    exist_ok    = False,
    # Disable colour augments — depth channel has no colour information
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    # Geometry augments (same as YOLO11n-RGBD run)
    degrees=20.0, translate=0.05, scale=0.25,
    shear=2.0, perspective=0.0001,
    flipud=0.0, fliplr=0.5,
    mosaic=0.2, erasing=0.2,
    auto_augment="randaugment",
)
