#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models/pose_landmarker_lite.task"
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
mkdir -p "$ROOT/models"
if [[ -f "$DEST" ]]; then
  echo "already have $DEST"
  exit 0
fi
curl -L --fail -o "$DEST" "$URL"
ls -lh "$DEST"
