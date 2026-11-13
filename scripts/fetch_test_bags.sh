#!/usr/bin/env bash
# fetch_test_bags.sh
#
# Download the CSSR4Africa RealSense test bags (ROS1) from Hugging Face and
# convert them to ROS2 bags for the face_detection and person_detection
# bag-replay tests.
#
#   Source: https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files
#   Output: ${PEPPER4DEC_TEST_BAGS:-~/ros2_ws/bags/cssr4africa}/<package>/<name>/
#
# The bags are written as rosbag2 metadata version 8: Humble cannot parse
# version 9 (the default) and rosbags writes nothing older. Only the colour
# stream (/camera/color/image_raw/compressed) is kept; the tests feed it to
# the nodes in RGB-only mode. The ROS1 .bag files are
# deleted after conversion unless --keep-ros1 is given. The converter
# (rosbags) is installed into its own venv, so nothing system-wide changes.
#
# Usage: scripts/fetch_test_bags.sh [--keep-ros1]
#
# Author: Yohannes Tadesse Haile
# Affiliation: Carnegie Mellon University Africa
# Date: November 13, 2026
# Version: v1.0

set -euo pipefail

HF=https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files/resolve/main
OUT=${PEPPER4DEC_TEST_BAGS:-$HOME/ros2_ws/bags/cssr4africa}
VENV=${ROSBAGS_VENV:-$HOME/ros2_ws/.venv_rosbags}
TOPIC=/camera/color/image_raw/compressed
KEEP_ROS1=false
[[ "${1:-}" == "--keep-ros1" ]] && KEEP_ROS1=true

BAGS=(
  face_detection/face_detection_test_input_realsense_single_face
  face_detection/face_detection_test_input_realsense_multiple_faces
  face_detection/face_detection_test_input_realsense_mutual_gaze
  face_detection/face_detection_test_input_realsense_lighting_1
  face_detection/face_detection_test_input_realsense_lighting_2
  person_detection/person_detection_test_input_realsense_single_person
  person_detection/person_detection_test_input_realsense_multiple_people
  person_detection/person_detection_test_input_realsense_lighting_1
  person_detection/person_detection_test_input_realsense_lighting_2
)

if [[ ! -x "$VENV/bin/rosbags-convert" ]]; then
  echo "Installing rosbags into $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q rosbags
fi

for entry in "${BAGS[@]}"; do
  pkg=${entry%%/*}
  name=${entry#*/}
  dst="$OUT/$pkg/${name#"${pkg}_test_input_realsense_"}"
  if [[ -f "$dst/metadata.yaml" ]]; then
    echo "have  $dst"
    continue
  fi
  mkdir -p "$OUT/ros1/$pkg" "$OUT/$pkg"
  src="$OUT/ros1/$pkg/$name.bag"
  if [[ ! -f "$src" ]]; then
    echo "fetch $pkg/$name.bag"
    curl -fsSL --retry 3 -o "$src.part" "$HF/${pkg}_test/data/$name.bag"
    mv "$src.part" "$src"
  fi
  echo "conv  $dst"
  rm -rf "$dst"
  "$VENV/bin/rosbags-convert" --src "$src" --dst "$dst" --include-topic "$TOPIC" --dst-version 8 >/dev/null
  $KEEP_ROS1 || rm -f "$src"
done
$KEEP_ROS1 || rmdir "$OUT"/ros1/* "$OUT/ros1" 2>/dev/null || true
echo "Test bags ready in $OUT"
