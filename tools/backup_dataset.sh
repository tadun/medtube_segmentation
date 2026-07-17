#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Backup dataset to Google Drive (or any external location)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The dataset/ folder is gitignored (too large for git). This script creates
# a compressed archive for safe backup.
#
# Usage:
#   ./tools/backup_dataset.sh                          # → backs up to ~/Google Drive
#   ./tools/backup_dataset.sh /Volumes/USB/backups     # → custom location
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$SCRIPT_DIR/dataset"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="medtube_dataset_backup_${TIMESTAMP}.tar.gz"

# Default backup location
BACKUP_DIR="${1:-$HOME/Library/CloudStorage/GoogleDrive/medtube_backups}"

if [ ! -d "$DATASET_DIR" ]; then
    echo -e "\033[31m[error]\033[0m dataset/ folder not found at $DATASET_DIR"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

echo -e "\033[36m[info]\033[0m Backing up dataset..."
echo "  Source:  $DATASET_DIR"
echo "  Target:  $BACKUP_DIR/$ARCHIVE_NAME"

# Count files
TOTAL=$(find "$DATASET_DIR" -name "*.png" | wc -l | tr -d ' ')
echo "  Files:   $TOTAL PNG images"

# Create compressed archive
tar -czf "$BACKUP_DIR/$ARCHIVE_NAME" -C "$SCRIPT_DIR" dataset/

SIZE=$(du -sh "$BACKUP_DIR/$ARCHIVE_NAME" | cut -f1)
echo -e "\033[32m[done]\033[0m Archive created: $SIZE"
echo ""
echo "To restore:"
echo "  cd $SCRIPT_DIR && tar -xzf $BACKUP_DIR/$ARCHIVE_NAME"
