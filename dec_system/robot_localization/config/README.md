# robot_localization config — odometry input

## odom_topic

Set to `/wheel_odom` — Pepper's wheel encoder odometry published by naoqi_driver2.

Do not change to `/odom`. The plain `/odom` topic is reserved for SLAM/EKF fusion output
and will conflict if another algorithm is running simultaneously.

## Related

- Source: `naoqi_driver2/doc/odometry_naming.md` — explains why the driver uses `/wheel_odom`
- Frame convention: `pepper_navmap/config/README.md`
