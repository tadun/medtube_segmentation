#!/bin/bash
# Launcher for RealSense scripts.
# Runs via 'sudo realsense-run' (passwordless, scoped to this binary only).
# Required on macOS because libusb needs root to claim the USB IOKit interface.
exec sudo /usr/local/bin/realsense-run "$@"
