#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# MedTube Segmentation — Demo Recording & Screenshot Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: Record a demonstration video and capture all key screenshots for the
#          final project report in a single session.
#
# SETUP:
#   1. Place tubes on the matte surface under the camera
#   2. Run: ./stream.sh
#   3. Follow the timeline below (screenshots = SPACE, recording = R)
#
# TUBES NEEDED:
#   - tube_2 (Universal — red cap)
#   - tube_1 (Screwcap — green label)
#   - tube_4 (Push-on — blue cap)
#   - tube_8 (Other — yellow label)
#   - tubes 5, 6, 7 (extra Push-on for multi-instance scene)
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── TIMELINE (approx 3 minutes) ─────────────────────────────────────────────
#
# [0:00] ./stream.sh — starts on YOLO26n
#        Wait for "Depth ROI locked" in terminal
#
# ─── SCREENSHOT 1: System overview (all 4 classes) ───────────────────────────
# [0:10] Place tube_2, tube_1, tube_4, tube_8 in view
# [0:15] SPACE — "Fig: all classes, YOLO26n"
#
# ─── SCREENSHOT 2: Best accuracy model ───────────────────────────────────────
# [0:20] M → switch to YOLOv8m (watch HUD show new model name)
# [0:25] SPACE — "Fig: all classes, YOLOv8m"
#
# ─── SCREENSHOT 3–4: Architecture comparison ─────────────────────────────────
# [0:30] M → YOLOv9c
# [0:35] SPACE — "Fig: all classes, YOLOv9c"
# [0:40] M → YOLO11n
# [0:45] SPACE — "Fig: all classes, YOLO11n"
#
# ─── RECORDING 1: Speed comparison across models ─────────────────────────────
# [0:50] M → back to YOLO26n
# [0:55] R — start recording
# [1:00] Keep single tube (Universal) visible
# [1:10] M (switch to YOLOv8m, watch FPS drop)
# [1:20] M (switch to YOLOv9c)
# [1:30] M (switch to YOLO11n)
# [1:40] R — stop recording
#
# ─── SCREENSHOT 5: Multi-instance detection ──────────────────────────────────
# [1:45] M → back to YOLO26n
# [1:50] Place 3–4 Push-on tubes (tubes 4, 5, 6, 7)
# [1:55] SPACE — "Fig: multi-instance"
#
# ─── SCREENSHOT 6: Hardest class (Screwcap) ──────────────────────────────────
# [2:00] Remove Push-on tubes, place 2–3 Screwcap tubes
# [2:05] SPACE — "Fig: Screwcap, YOLO26n"
# [2:10] M → YOLOv8m
# [2:15] SPACE — "Fig: Screwcap, YOLOv8m (expect better accuracy)"
#
# ─── SCREENSHOT 7: Depth heatmap ─────────────────────────────────────────────
# [2:20] Keep tubes in place (any class)
#        SPACE — "Fig: depth heatmap + mask overlay (TR + BR panels)"
#
# ─── SCREENSHOT 8: Empty scene (no false positives) ──────────────────────────
# [2:25] Remove ALL tubes from surface
# [2:30] SPACE — "Fig: clean scene, no detections"
#
# ─── END ─────────────────────────────────────────────────────────────────────
# [2:35] Q — quit
#        Session summary prints in terminal (capture terminal screenshot too)
#
# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUTS:
#   runs/captures/snapshots/  — 8 snapshots (4 PNGs each = 32 files)
#   runs/captures/rec_*/      — 1 recording (speed comparison video)
#   Terminal screenshot        — session summary with model names + FPS
#
# REPORT FIGURES FROM THIS SESSION:
#   Fig 1: System overview (4 classes, YOLO26n) — snapshot 1
#   Fig 2: Model comparison (4 snapshots, same scene, different models)
#   Fig 3: Multi-instance detection — snapshot 5
#   Fig 4: Screwcap comparison (26n vs v8m) — snapshots 6–7
#   Fig 5: Depth integration (heatmap + masks) — snapshot 8 (TR+BR panels)
#   Fig 6: Empty scene validation — snapshot 9
#   Fig 7: Speed/FPS comparison — recording 1 (extract key frames)
# ═══════════════════════════════════════════════════════════════════════════════
