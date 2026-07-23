cmake_minimum_required(VERSION 3.8)
project(dec_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python nodes (RealSense depth ROI service, LiDAR colorizer/calibrator)
install(PROGRAMS
  scripts/depth_roi_service.py
  scripts/lidar_colorizer.py
  scripts/lidar_depth_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
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