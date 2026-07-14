#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RealSense stream launcher  —  passwordless sudo on macOS
# ─────────────────────────────────────────────────────────────────────────────
# macOS libusb/librealsense needs root to claim the USB interface.
# Run ONE-TIME SETUP once, then this script never asks for a password:
#
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   sudo ln -sf "$SCRIPT_DIR/rs_env/bin/python" /usr/local/bin/rs-python-medtube
#   echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/local/bin/rs-python-medtube" \
#       | sudo tee /etc/sudoers.d/realsense
# ─────────────────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LAUNCHER=/usr/local/bin/rs-stream-medtube

if [ ! -f "$LAUNCHER" ]; then
    echo "[setup needed] Run the ONE-TIME SETUP from this script's header, then rerun:"
    echo "  cat > /tmp/rs-stream-medtube << 'WEOF'"
    echo "  #!/bin/bash"
    echo "  PROJ=\"$SCRIPT_DIR\""
    echo "  exec \"\$PROJ/rs_env/bin/python\" \"\$PROJ/realsense_stream.py\" \"\$@\""
    echo "  WEOF"
    echo "  sudo install -m 755 /tmp/rs-stream-medtube $LAUNCHER"
    echo "  echo \"$(whoami) ALL=(ALL) NOPASSWD: $LAUNCHER\" | sudo tee /etc/sudoers.d/realsense"
    echo "  sudo chmod 440 /etc/sudoers.d/realsense"
    exit 1
fi

exec sudo "$LAUNCHER" "$@"
