cmake_minimum_required(VERSION 3.8)
project(pepper_slam)

# Launch/params only - no compiled nodes. Every SLAM component (RTAB-Map,
# SLAM Toolbox, FAST-LIO) is an upstream package launched by name, so this
# package needs no find_package() beyond ament_cmake itself; the runtime
# requirements are declared as exec_depend in package.xml.
find_package(ament_cmake REQUIRED)

# One plain install(DIRECTORY), matching FAST_LIO/CMakeLists.txt:132-135.
#
# WHY NOT FILES_MATCHING (2026-08-10): these were four separate blocks each
# using `FILES_MATCHING PATTERN "*.py"` / "*.yaml" / "*.xacro" / "*.rviz".
# colcon's --symlink-install cannot symlink through a FILES_MATCHING filter --
# it silently falls back to COPYING -- so pepper_slam alone among the SLAM
# packages installed real copies while fast_lio, point_lio and fastlio_lc_pgo
# installed symlinks. That divergence is not cosmetic: config/sensor_tf.yaml
# was edited in install/ on 2026-08-06 (the measured camera rotation) and the
# correction survived ONLY there, three days after src/ had been reverted to
# an older value. A routine rebuild would have destroyed the only copy.
# With symlinks, install/ and src/ cannot drift apart at all.
#
# Dropping the patterns also subsumes the old note about urdf/ needing both
# "*.urdf" and "*.xacro": the whole directory is installed, so the top-level
# *.urdf.xacro descriptions and the shared sensor_rig.xacro macro they include
# both resolve at launch.
install(DIRECTORY
  launch config urdf rviz
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
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
