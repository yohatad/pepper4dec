cmake_minimum_required(VERSION 3.8)
project(pepper_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(nav2_bringup REQUIRED)
find_package(nav2_map_server REQUIRED)
find_package(nav2_amcl REQUIRED)
find_package(nav2_controller REQUIRED)
find_package(nav2_planner REQUIRED)
find_package(nav2_behaviors REQUIRED)
find_package(nav2_bt_navigator REQUIRED)
find_package(nav2_lifecycle_manager REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nav2_msgs REQUIRED)

# Dev/debug tooling (src/tools/) - not part of the production navigation
# pipeline. pepper_odom_covariance now lives in its own top-level package.

# odom_path_publisher: accumulates /pepper_odom into a nav_msgs/Path +
# RViz markers for closed-loop odometry quality testing
add_executable(odom_path_publisher
  src/tools/odom_path_publisher.cpp
)
ament_target_dependencies(odom_path_publisher
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_srvs
)
install(TARGETS odom_path_publisher
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# send_goal: one-shot Nav2 goal sender (reimplements the relevant subset of
# nav2_simple_commander's Python-only BasicNavigator)
add_executable(send_goal
  src/tools/send_goal.cpp
)
ament_target_dependencies(send_goal
  rclcpp
  rclcpp_action
  geometry_msgs
  nav2_msgs
)
install(TARGETS send_goal
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Note: tools/generate_keepout.py is a standalone map-editing utility, not a
# ROS2 node - run manually with `python3 tools/generate_keepout.py`, not
# installed here.

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

# Install rviz configs
install(DIRECTORY
  rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# Install map files
install(DIRECTORY
  map/
  DESTINATION share/${PROJECT_NAME}/map
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.pgm"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  # flake8 is run manually below so rtabmap_realsense.launch.py (vendored
  # from an upstream rtabmap_ros example, not our style) can be excluded
  set(ament_cmake_flake8_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ament_flake8.ini")
endif()

ament_package()
