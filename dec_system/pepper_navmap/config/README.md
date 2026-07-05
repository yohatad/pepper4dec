# pepper_navmap config — odometry naming

## Frame and topic convention

Pepper's wheel odometry uses distinct names to avoid conflicts with SLAM/VIO/LIO:

| What | Name |
|------|------|
| Wheel odom topic (from naoqi_driver2) | `/pepper_odom` |
| Wheel odom TF frame | `pepper_odom` |
| Fused/SLAM odom frame (Nav2 primary) | `odom` |

## How it maps to these config files

**`nav2_params.yaml`**
- `amcl.odom_frame_id: "pepper_odom"` — AMCL expects Pepper's wheel odom TF
- `bt_navigator.odom_topic: pepper_odom` — BT navigator subscribes to wheel odom topic
- `local_costmap.global_frame: odom` — local costmap anchors to the fused odom frame (EKF output); update this to `pepper_odom` if the EKF is removed and the driver's TF is used directly

**`ekf_nav.yaml.yaml`**
- `odom0: /pepper_odom` — EKF input: raw wheel odometry from the driver
- `odom_frame: pepper_odom` / `world_frame: pepper_odom` — EKF output frame

**`mapper_params_online_async.yaml`**
- `odom_frame: pepper_odom` — slam_toolbox looks for `pepper_odom → base_footprint` TF

## Do not change these back to "odom"

The plain `odom` name is used by SLAM outputs (Point-LIO, RTAB-Map, etc.).
If the driver's wheel odom is renamed back to `odom`, any running SLAM system will silently overwrite it.
