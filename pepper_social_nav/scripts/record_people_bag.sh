#!/usr/bin/env bash
# Record the validation bag the people tracker is developed against.
#
# WHY A NEW BAG. The existing recordings (slam_recording*, slam_bench_run*,
# lidar_cam_calib*) are SLAM runs: mostly empty corridors, with people in shot
# only by accident. Nothing in them exercises two-camera association, a head
# pan across a person, or a walking velocity estimate. Without a bag that does,
# "validate the tracker on bag replay" is not a gate you can actually pass.
#
# WHAT TO CAPTURE. Run each of these for ~30 s, with the robot STATIONARY (the
# tracker is being tested, not the localizer):
#   1. One person walking left-to-right across the front camera at ~1 m/s.
#   2. One person walking straight at the robot, stopping at ~1 m.
#   3. One person standing still for 20 s        -> must report ~0 velocity.
#   4. Two people crossing paths                 -> the ID-swap case.
#   5. A person leaving frame and returning       -> the case ReID would claim;
#      the world-frame filter should bridge it within max_track_age.
#   6. overt_attention panning the head across a standing person
#      -> the ID-churn case this whole design exists to remove.
#   7. A person at 1 m, 2 m, 3 m in turn -> confirms the RealSense's vertical
#      framing limit (waist at 2 m) and that ground-plane ranging still works
#      when the head is cropped.
#
# Then replay with:
#   ros2 bag play <bag> --clock
#   ros2 launch pepper_social_nav perception_offboard.launch.py use_sim_time:=true
#
# /tf_static is recorded transient_local via pepper_slam's record_qos.yaml --
# without that override the recorder races the publishers and captures ZERO
# static transforms, and the tracker then cannot deproject anything at all.
set -euo pipefail

OUT="${1:-people_bag_$(date +%Y%m%d_%H%M%S)}"
QOS="$(ros2 pkg prefix pepper_slam)/share/pepper_slam/config/record_qos.yaml"

if [[ ! -f "$QOS" ]]; then
  echo "error: record_qos.yaml not found at $QOS (is pepper_slam built and sourced?)" >&2
  exit 1
fi

echo "Recording to '$OUT'. Ctrl-C to stop."
echo "Keep the robot stationary; move the PEOPLE, not the base."

# Compressed image topics only: the raw streams are far too large to sustain,
# and the offboard pipeline consumes the compressed ones anyway.
exec ros2 bag record \
  --qos-profile-overrides-path "$QOS" \
  -o "$OUT" \
  /tf /tf_static \
  /joint_states \
  /camera/color/image_raw/compressed \
  /camera/color/camera_info \
  /camera/aligned_depth_to_color/image_raw/compressedDepth \
  /camera/aligned_depth_to_color/camera_info \
  /camera/front/image_raw/compressed \
  /camera/front/camera_info \
  /points \
  /imu/data \
  /odom_lio
