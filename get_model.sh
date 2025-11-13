#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_FILE="$TARGET_DIR/fer2013_model.pth"

echo "[INFO] This script is a template. Replace the URL below with your hosting link."
MODEL_URL="https://example.com/path/to/fer2013_model.pth"

read -p "Download weights to $TARGET_FILE ? [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  echo "[INFO] Downloading weights..."
  curl -fL "$MODEL_URL" -o "$TARGET_FILE"
  echo "[OK] Saved to $TARGET_FILE"
else
  echo "[SKIP] Download canceled. Place the file manually at:"
  echo "       $TARGET_FILE"
fi




