#!/usr/bin/env bash
# fetch_models.sh
#
# Download the ONNX models that face_detection, person_detection and
# speech_event load into their (git-ignored) models/ directories, and verify
# their SHA-256 checksums.
#
#   Source: https://huggingface.co/cssr4africa/dec_models
#           (<package>/models/<file>, as in cssr4africa/cssr4africa_models)
#
# Licences differ per model (YOLO11m AGPL-3.0, Gold-YOLO GPL-3.0); see
# MODELS.md. If the repository is private, export PEPPER4DEC_HF_TOKEN (a
# read token for cssr4africa).
# The MiVOLO age/gender model (age_gender_detection) is not in the repository
# yet and must be placed by hand.
#
# Usage: scripts/fetch_models.sh
#
# Author: Yohannes Tadesse Haile
# Affiliation: Carnegie Mellon University Africa
# Date: November 14, 2026
# Version: v1.0
#
# Copyright (C) 2025 Carnegie Mellon University Africa

set -euo pipefail

REPO=${PEPPER4DEC_MODELS_REPO:-cssr4africa/dec_models}
HF=${PEPPER4DEC_MODELS_URL:-https://huggingface.co/$REPO/resolve/main}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AUTH=()
# Only this variable, not HF_TOKEN: that may hold a token for another account.
[[ -n "${PEPPER4DEC_HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $PEPPER4DEC_HF_TOKEN")

# <sha256> <package>/models/<file>
MODELS=(
  "d2fd2ecaa7326b6684cd379897a9d00f30ae27a59cad274ce54e59cc246b22a1 person_detection/models/person_detection_yolov11m.onnx"
  "10b1c58fb46320fb73ab4e95104852a21c5a41ee64372b49060e580504e29881 face_detection/models/face_detection_goldYOLO.onnx"
  "221adb6688f5d1df5c9937e3e5e473bb84729a0853151c6429d8528ccd8b18b1 face_detection/models/face_detection_sixdrepnet360.onnx"
  "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3 speech_event/models/silero_vad.onnx"
)

for entry in "${MODELS[@]}"; do
  sha=${entry%% *}
  path=${entry#* }
  dst="$ROOT/$path"
  if [[ -f "$dst" ]] && echo "$sha  $dst" | sha256sum -c --status -; then
    echo "have  $path"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  echo "fetch $path"
  for attempt in 1 2 3 4 5; do
    curl -fsSL --retry 3 -C - "${AUTH[@]}" -o "$dst.part" "$HF/$path" && break
    [[ $attempt == 5 ]] && { echo "download failed: $HF/$path" >&2; exit 1; }
    sleep 5
  done
  if ! echo "$sha  $dst.part" | sha256sum -c --status -; then
    rm -f "$dst.part"
    echo "checksum mismatch: $path" >&2
    exit 1
  fi
  mv "$dst.part" "$dst"
done
echo "Models ready under $ROOT"
