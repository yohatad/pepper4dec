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
this repository (`pepper4dec`) - it's infrastructure-tier, alongside `naoqi_driver2`, not
navigation-specific.

## How it maps to these config files

**`nav2_params.yaml`**
- `amcl.odom_frame_id: "pepper_odom"` — AMCL expects Pepper's wheel odom TF
- `bt_navigator.odom_topic: pepper_odom_filtered` — BT navigator subscribes to the covariance-corrected wheel odom topic
- `local_costmap.global_frame: pepper_odom` — local costmap tracks the live wheel-odom TF directly. Previously set to `odom`, a frame nothing published (neither the driver nor the unlaunched EKF ever broadcasts anything named plain `odom` - the EKF's own `world_frame`/`odom_frame` are `pepper_odom`, see below). Switch this to `odom` once a genuine SLAM/LIO source is actually publishing that frame.

**`ekf_nav.yaml.yaml`**
- `odom0: /pepper_odom_filtered` — EKF input: wheel odometry, covariance-corrected by `pepper_odom_covariance`
- `odom_frame: pepper_odom` / `world_frame: pepper_odom` — EKF output frame (deliberately not `odom` - that name is reserved for a future SLAM/LIO source, see below)
- `base_link_frame: base_footprint` — must match the driver's actual child_frame_id (`base_footprint`); a previous `base_footprint_nav` typo here meant the EKF would have published a transform to a frame disconnected from the rest of the TF tree

**`mapper_params_online_async.yaml`** — moved to the `pepper_slam` package
(`pepper_slam/config/`) along with the rest of the SLAM stack. It follows the
same convention: `odom_frame: pepper_odom`, i.e. slam_toolbox looks for the
`pepper_odom → base_footprint` TF. The naming rules on this page apply to both
packages.

## Do not change these back to "odom"

The plain `odom` name is used by SLAM outputs (Point-LIO, RTAB-Map, etc.).
If the driver's wheel odom is renamed back to `odom`, any running SLAM system will silently overwrite it.
