#!/usr/bin/env bash
# install_onnxruntime_gpu.sh [DEST]
#
# Download the official ONNX Runtime GPU release that face_detection and
# person_detection build against, verify its checksum and unpack it into DEST
# (default: ~/ros2_ws/.deps). dec_common's dec_find_onnxruntime() finds it
# there without further setup; rebuild the two packages afterwards:
#
#   colcon build --packages-select dec_common face_detection person_detection \
#     --cmake-force-configure
#
# ROS's onnxruntime_vendor is CPU-only; it remains the fallback when this
# release is absent. Keep the release out of /opt/ros: an apt upgrade there
# replaces the core library and leaves a mismatched CUDA provider behind.
#
# Also used by the Dockerfile.
#
# Author: Yohannes Tadesse Haile
# Affiliation: Carnegie Mellon University Africa
# Date: November 14, 2026
# Version: v1.0
#
# Copyright (C) 2025 Carnegie Mellon University Africa

set -euo pipefail

VERSION=1.23.2
NAME=onnxruntime-linux-x64-gpu-${VERSION}
SHA256=2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53
URL=https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${NAME}.tgz
DEST=${1:-$HOME/ros2_ws/.deps}

if [ -f "$DEST/$NAME/lib/libonnxruntime.so" ]; then
  echo "ONNX Runtime $VERSION (GPU) already in $DEST/$NAME"
  exit 0
fi

mkdir -p "$DEST"
tmp=$(mktemp -d "$DEST/.onnxruntime-dl.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $URL (about 240 MB)"
# -C - resumes a partial download on retry.
for attempt in 1 2 3 4 5; do
  if curl -fL --retry 3 --no-progress-meter -C - -o "$tmp/$NAME.tgz" "$URL"; then break; fi
  [ "$attempt" = 5 ] && { echo "download failed" >&2; exit 1; }
  sleep 5
done
echo "$SHA256  $tmp/$NAME.tgz" | sha256sum -c -

tar xzf "$tmp/$NAME.tgz" -C "$tmp"
# The TensorRT provider is not used and is the largest file in the release.
rm -f "$tmp/$NAME/lib/libonnxruntime_providers_tensorrt.so"
mv "$tmp/$NAME" "$DEST/$NAME"
echo "Installed $DEST/$NAME"
