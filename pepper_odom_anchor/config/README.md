# pepper_odom_anchor config — odometry input

## odom_topic

Set to `/pepper_odom_filtered` — not naoqi_driver2's raw output directly. naoqi_driver2
publishes wheel+IMU odometry on `/pepper_odom` with a flat, non-growing covariance;
`pepper_odom_covariance` subscribes to that and republishes as `/pepper_odom_filtered`
with a covariance that grows with distance/rotation traveled, which is what this node
(and the EKF) should consume instead.

Do not change to `/pepper_odom`. That topic is naoqi_driver2's raw output and will
conflict with the improved-covariance pipeline if consumed directly.

## Related

- Covariance pipeline: `pepper_odom_covariance` package (top-level,
  `~/ros2_ws/src/pepper_odom_covariance/` - not under `dec_system`)
- Frame convention: `pepper_navigation/config/README.md`
