cmake_minimum_required(VERSION 3.8)
project(pepper_slam)

# Launch/params only - no compiled nodes. Every SLAM component (RTAB-Map,
# SLAM Toolbox, FAST-LIO) is an upstream package launched by name, so this
# package needs no find_package() beyond ament_cmake itself; the runtime
# requirements are declared as exec_depend in package.xml.
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# Both patterns are needed: the top-level descriptions are *.urdf.xacro and the
# shared macro they include is sensor_rig.xacro, so "*.urdf" alone installs
# nothing and $(find pepper_slam)/urdf/... fails to resolve at launch.
install(DIRECTORY
  urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
  FILES_MATCHING PATTERN "*.urdf" PATTERN "*.xacro"
)

install(DIRECTORY
  rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

install(PROGRAMS
  scripts/leveled_odometry_publisher.py
  scripts/pepper_odom_relabel.py
  scripts/check_frame_contract.py
  scripts/static_tf_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # flake8 is run manually below so rtabmap_base.launch.py (vendored
  # from an upstream rtabmap_ros example, not our style) can be excluded.
  # ament_cmake_flake8_FOUND doesn't suppress the auto-discovered hook here
  # (its extension gets registered before that variable takes effect) -
  # AMENT_LINT_AUTO_EXCLUDE is the mechanism ament_lint_auto actually
  # documents for "run this linter manually instead of automatically".
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_flake8)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ament_flake8.ini")
endif()

ament_package()
