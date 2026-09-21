#!/usr/bin/env bash
# Download the models.
#   ./scripts/fetch_model.sh              face landmarks only (~3.8 MB)
#   ./scripts/fetch_model.sh --identity   also the face recogniser (~37 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

LANDMARKS="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SFACE="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

curl -fL --progress-bar "$LANDMARKS" -o models/face_landmarker.task

if [ "${1:-}" = "--identity" ]; then
    curl -fL --progress-bar "$SFACE" -o models/face_recognition_sface.onnx
fi

ls -lh models/
