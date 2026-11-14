#!/usr/bin/env bash
# fetch_test_bags.sh
#
# Download the CSSR4Africa RealSense test bags (ROS2) from Hugging Face for
# the face_detection and person_detection bag-replay tests.
#
#   Source: https://huggingface.co/cssr4africa/dec_unit_test_data_files
#           (ROS2 copies of https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files,
#           made by scripts/convert_cssr4africa_bags.sh)
#   Output: ${PEPPER4DEC_TEST_BAGS:-~/ros2_ws/bags/cssr4africa}/<package>/<name>/
#
# Each bag holds every recorded topic (colour, aligned depth, camera_info,
# /tf_static), 15 fps, rosbag2 metadata version 8 (readable by Humble). The
# person_detection recordings are byte-identical to face_detection ones, so
# each is downloaded once (about 370 MB in all) and copied. If the dataset is
# private, export PEPPER4DEC_HF_TOKEN (a read token for cssr4africa).
#
# Usage: scripts/fetch_test_bags.sh
#
# Author: Yohannes Tadesse Haile
# Affiliation: Carnegie Mellon University Africa
# Date: November 14, 2026
# Version: v2.0

set -euo pipefail

REPO=${PEPPER4DEC_BAGS_REPO:-cssr4africa/dec_unit_test_data_files}
HF=${PEPPER4DEC_BAGS_URL:-https://huggingface.co/$REPO/resolve/main}
OUT=${PEPPER4DEC_TEST_BAGS:-$HOME/ros2_ws/bags/cssr4africa}
AUTH=()
# Only this variable, not HF_TOKEN: that may hold a token for another account.
[[ -n "${PEPPER4DEC_HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $PEPPER4DEC_HF_TOKEN")

# Local names; the repository names add <package>_test_input_realsense_.
FACE=(single_face multiple_faces mutual_gaze lighting_1 lighting_2)
declare -A PERSON=([single_person]=single_face [multiple_people]=multiple_faces
                   [lighting_1]=lighting_1 [lighting_2]=lighting_2)

get() {  # get <repository path> <local file>
  [[ -f "$2" ]] && return
  for attempt in 1 2 3 4 5; do
    curl -fsSL --retry 3 -C - "${AUTH[@]}" -o "$2.part" "$HF/$1" && break
    [[ $attempt == 5 ]] && { echo "download failed: $HF/$1" >&2; exit 1; }
    sleep 5
  done
  mv "$2.part" "$2"
}

for n in "${FACE[@]}"; do
  dst="$OUT/face_detection/$n"
  if [[ -f "$dst/metadata.yaml" ]] && grep -q compressedDepth "$dst/metadata.yaml"; then
    echo "have  $dst"
    continue
  fi
  # Replace an older colour-only conversion.
  rm -rf "$dst"
  mkdir -p "$dst.part"
  name=face_detection_test_input_realsense_$n
  echo "fetch $name"
  get "face_detection_test/data/$name/$name.db3" "$dst.part/$name.db3"
  get "face_detection_test/data/$name/metadata.yaml" "$dst.part/metadata.yaml"
  mv "$dst.part" "$dst"
done

for p in "${!PERSON[@]}"; do
  dst="$OUT/person_detection/$p"
  src="$OUT/face_detection/${PERSON[$p]}"
  if [[ -f "$dst/metadata.yaml" ]] && grep -q compressedDepth "$dst/metadata.yaml"; then
    echo "have  $dst"
    continue
  fi
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -rl "$src" "$dst"  # hard links: no second copy on disk
  echo "copy  $dst (= ${PERSON[$p]})"
done
echo "Test bags ready in $OUT"
