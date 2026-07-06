# pepper_navigation config — odometry naming

## Frame and topic convention

Pepper's wheel odometry uses distinct names to avoid conflicts with SLAM/VIO/LIO:

| What | Name |
|------|------|
| Wheel odom topic (naoqi_driver2's raw output, flat covariance) | `/pepper_odom` |
| Wheel odom topic, covariance-corrected by `pepper_odom_covariance` | `/pepper_odom_filtered` |
| Wheel odom TF frame (used by both topics above) | `pepper_odom` |
| Fused/SLAM odom frame (Nav2 primary) | `odom` |

`pepper_odom_covariance` is a separate top-level package
(`~/ros2_ws/src/pepper_odom_covariance/`), not part of `pepper_navigation` or
`dec_system` - it's infrastructure-tier, alongside `naoqi_driver2`, not
navigation-specific.

## How it maps to these config files

**`nav2_params.yaml`**
- `amcl.odom_frame_id: "pepper_odom"` — AMCL expects Pepper's wheel odom TF
- `bt_navigator.odom_topic: pepper_odom_filtered` — BT navigator subscribes to the covariance-corrected wheel odom topic
- `local_costmap.global_frame: odom` — local costmap anchors to the fused odom frame (EKF output); update this to `pepper_odom` if the EKF is removed and the driver's TF is used directly

**`ekf_nav.yaml.yaml`**
- `odom0: /pepper_odom_filtered` — EKF input: wheel odometry, covariance-corrected by `pepper_odom_covariance`
- `odom_frame: pepper_odom` / `world_frame: pepper_odom` — EKF output frame

**`mapper_params_online_async.yaml`**
- `odom_frame: pepper_odom` — slam_toolbox looks for `pepper_odom → base_footprint` TF

## Do not change these back to "odom"

The plain `odom` name is used by SLAM outputs (Point-LIO, RTAB-Map, etc.).
If the driver's wheel odom is renamed back to `odom`, any running SLAM system will silently overwrite it.
