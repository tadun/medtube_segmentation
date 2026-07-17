#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# MedTube Segmentation — Live Demo Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: Demonstrate real-time instance segmentation of medical tubes using
#          the Intel RealSense D415 depth camera and multiple YOLO architectures.
#
# SETUP:
#   1. Place tubes on the matte surface under the camera
#   2. Run: ./stream.sh
#   3. Follow the sequence below
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── DEMO SEQUENCE (approx 5 minutes total) ──────────────────────────────────

# SCENE 1: Single tube per class (60s)
# Place one tube of each class in view simultaneously:
#   - tube_2 (Universal) — red cap
#   - tube_1 (Screwcap) — green label
#   - tube_4 (Push-on) — blue cap
#   - tube_8 (Other) — yellow label
#
# Actions:
#   [0:00] Start stream (./stream.sh) — model starts as yolo26n-seg
#   [0:05] Wait for depth ROI lock + first detections
#   [0:10] Press SPACE — snapshot all 4 classes detected simultaneously
#   [0:15] Press M — switch to yolov8m-seg (note FPS drop in HUD)
#   [0:20] Press SPACE — snapshot for comparison
#   [0:25] Press M — switch to yolov9c-seg
#   [0:30] Press SPACE — snapshot
#   [0:35] Press M — switch to yolo11n-seg
#   [0:40] Press SPACE — snapshot
#   [0:45] Press M — back to yolo26n-seg

# SCENE 2: Multiple tubes of same class (45s)
# Place 3–4 Push-on tubes (tubes 4, 5, 6, 7) spread across the surface
#
# Actions:
#   [0:50] Arrange Push-on tubes
#   [0:55] Press SPACE — multi-instance detection
#   [1:00] Press R — start recording
#   [1:10] Slowly move/rotate one tube (shows tracking continuity)
#   [1:15] Press R — stop recording

# SCENE 3: Mixed classes overlapping (45s)
# Stack tubes partially overlapping (hardest case):
#   - tube_1 (Screwcap) crossing over tube_4 (Push-on)
#   - tube_8 (Other) next to tube_2 (Universal)
#
# Actions:
#   [1:20] Arrange overlapping tubes
#   [1:30] Press SPACE — capture overlapping detection
#   [1:35] Press M — cycle through all models capturing SPACE for each

# SCENE 4: Screwcap confusion test (30s)
# Place only Screwcap tubes (tube_1 variants) — this is the hardest class
#
# Actions:
#   [2:05] Place 2–3 Screwcap tubes
#   [2:10] Press SPACE on yolo26n
#   [2:15] Press M → SPACE on yolov8m (expected: better Screwcap accuracy)

# SCENE 5: Speed comparison (60s)
# Single tube, cycle through all models watching FPS in HUD:
#   yolo26n (~7 FPS) → yolov8m (~3 FPS) → yolov9c (~2 FPS) → yolo11n (~7 FPS)
#
# Actions:
#   [2:35] Place one Universal tube in center
#   [2:40] Press R — start recording
#   [2:45] Press M every 10s (4 models × 10s = 40s)
#   [3:25] Press R — stop recording

# SCENE 6: Empty scene (15s)
# Remove all tubes — verify no false positives on bare matte surface
#
# Actions:
#   [3:30] Clear all tubes
#   [3:35] Press SPACE — should show clean panels with no detections

# END: Press Q to quit. Session summary prints in terminal.

# ═══════════════════════════════════════════════════════════════════════════════
# TOTAL TIME: ~4 minutes
# OUTPUTS:
#   - 8–10 snapshots in runs/captures/snapshots/
#   - 2 recordings in runs/captures/rec_*/
#   - Terminal session summary
# ═══════════════════════════════════════════════════════════════════════════════
