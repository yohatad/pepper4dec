#!/usr/bin/env bash
# convert_cssr4africa_bags.sh [OUT]
#
# Maintainer script: build the ROS2 copy of the CSSR4Africa RealSense test
# bags that scripts/fetch_test_bags.sh downloads. Users do not need it.
#
#   Source (ROS1): https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files
#   Target (ROS2): https://huggingface.co/cssr4africa/dec_unit_test_data_files
#
# Downloads the nine ROS1 bags into OUT/ros1 and writes OUT/ros2, which mirrors
# the ROS1 repository's layout (<package>_test/data/<name>/ as rosbag2
# directories) plus a dataset card, ready for:
#
#   hf upload cssr4africa/dec_unit_test_data_files OUT/ros2 .
#
# Every topic is kept (colour, aligned depth, both camera_info, /tf_static).
# The bags are written as rosbag2 metadata version 8, sqlite3 storage: Humble
# cannot parse version 9 and rosbags writes nothing older. The four
# person_detection bags are byte-identical to face_detection ones in the
# source, so each recording is converted once and copied under both names.
# The converter (rosbags) is installed into its own venv.
#
# Author: Yohannes Tadesse Haile
# Affiliation: Carnegie Mellon University Africa
# Date: November 14, 2026
# Version: v1.0
#
# Copyright (C) 2025 Carnegie Mellon University Africa

set -euo pipefail

SRC=https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files/resolve/main
OUT=${1:-$HOME/ros2_ws/bags/staging_cssr4africa_ros2}
VENV=${ROSBAGS_VENV:-$HOME/ros2_ws/.venv_rosbags}

# Recordings, and the person_detection names that are copies of them.
FACE=(single_face multiple_faces mutual_gaze lighting_1 lighting_2)
declare -A PERSON=([single_person]=single_face [multiple_people]=multiple_faces
                   [lighting_1]=lighting_1 [lighting_2]=lighting_2)

if [[ ! -x "$VENV/bin/rosbags-convert" ]]; then
  echo "Installing rosbags into $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q rosbags
fi

fetch() {  # fetch <path in the source repo>
  local dst="$OUT/ros1/$1"
  [[ -f "$dst" ]] && return
  mkdir -p "$(dirname "$dst")"
  echo "fetch $1"
  for attempt in 1 2 3 4 5; do
    curl -fsSL --retry 3 -C - -o "$dst.part" "$SRC/$1" && break
    [[ $attempt == 5 ]] && { echo "download failed: $1" >&2; exit 1; }
    sleep 5
  done
  mv "$dst.part" "$dst"
}

rm -rf "$OUT/ros2"
for n in "${FACE[@]}"; do
  name=face_detection_test_input_realsense_$n
  fetch "face_detection_test/data/$name.bag"
  dst="$OUT/ros2/face_detection_test/data/$name"
  mkdir -p "$(dirname "$dst")"
  echo "conv  $name"
  "$VENV/bin/rosbags-convert" --src "$OUT/ros1/face_detection_test/data/$name.bag" \
    --dst "$dst" --dst-version 8 > /dev/null
done

for p in "${!PERSON[@]}"; do
  name=person_detection_test_input_realsense_$p
  face=face_detection_test_input_realsense_${PERSON[$p]}
  # Check the claim that the source bags are identical before copying.
  fetch "person_detection_test/data/$name.bag"
  cmp -s "$OUT/ros1/person_detection_test/data/$name.bag" \
    "$OUT/ros1/face_detection_test/data/$face.bag" ||
    { echo "$name.bag differs from $face.bag; convert it separately" >&2; exit 1; }
  src="$OUT/ros2/face_detection_test/data/$face"
  dst="$OUT/ros2/person_detection_test/data/$name"
  mkdir -p "$dst"
  cp "$src/$face.db3" "$dst/$name.db3"
  sed "s/$face/$name/g" "$src/metadata.yaml" > "$dst/metadata.yaml"
  echo "copy  $name (= $face)"
done

cat > "$OUT/ros2/README.md" <<'EOF'
---
pretty_name: DEC unit test data files (ROS 2)
tags:
- robotics
- ros2
- rosbag2
- realsense
---

# DEC unit test data files (ROS 2)

ROS 2 copies of the Intel RealSense D435i recordings in
[cssr4africa/cssr4africa_unit_tests_data_files](https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files)
(ROS 1 bags), used by the face and person detection tests of
[pepper4dec](https://github.com/yohatad/pepper4dec), the ROS 2 Pepper tour
guide for the Upanzi Digital Experience Center at Carnegie Mellon University
Africa. The layout mirrors the ROS 1 repository: each bag is a rosbag2
directory (`metadata.yaml` plus one sqlite3 `.db3` file) at the path of the
ROS 1 `.bag` file without the extension.

| Bag | Duration | Frames |
|---|---|---|
| `face_detection_test/data/face_detection_test_input_realsense_single_face` | 38.5 s | 577 |
| `face_detection_test/data/face_detection_test_input_realsense_multiple_faces` | 27.8 s | 416 |
| `face_detection_test/data/face_detection_test_input_realsense_mutual_gaze` | 24.9 s | 373 |
| `face_detection_test/data/face_detection_test_input_realsense_lighting_1` | 43.2 s | 649 |
| `face_detection_test/data/face_detection_test_input_realsense_lighting_2` | 28.5 s | 427 |
| `person_detection_test/data/person_detection_test_input_realsense_single_person` | = single_face | |
| `person_detection_test/data/person_detection_test_input_realsense_multiple_people` | = multiple_faces | |
| `person_detection_test/data/person_detection_test_input_realsense_lighting_1` | = lighting_1 | |
| `person_detection_test/data/person_detection_test_input_realsense_lighting_2` | = lighting_2 | |

The four person detection bags are byte-identical to the face detection bags
they are marked with, as they are in the ROS 1 repository.

## Topics

| Topic | Type |
|---|---|
| `/camera/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` (JPEG, 640x480, 15 fps) |
| `/camera/aligned_depth_to_color/image_raw/compressedDepth` | `sensor_msgs/msg/CompressedImage` (`16UC1; compressedDepth png`, aligned to colour) |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/tf_static` | `tf2_msgs/msg/TFMessage` (transient local) |

## Conversion

Converted with [rosbags](https://gitlab.com/ternaris/rosbags)
(`rosbags-convert --dst-version 8`), keeping every topic and message
unchanged. Metadata version 8 is the newest that ROS 2 Humble reads.
The script that produced this repository is
[`scripts/convert_cssr4africa_bags.sh`](https://github.com/yohatad/pepper4dec/blob/devel/scripts/convert_cssr4africa_bags.sh)
in pepper4dec.

```bash
ros2 bag info face_detection_test/data/face_detection_test_input_realsense_single_face
ros2 bag play face_detection_test/data/face_detection_test_input_realsense_single_face
```

The recordings show people and are provided for testing the CSSR4Africa and
pepper4dec software, on the same terms as the ROS 1 originals.
EOF

du -sh "$OUT/ros2"
echo "ROS2 bags ready in $OUT/ros2"
