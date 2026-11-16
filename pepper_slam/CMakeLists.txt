cmake_minimum_required(VERSION 3.8)
project(pepper_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Every SLAM component (RTAB-Map, SLAM Toolbox, FAST-LIO) is an upstream
# package launched by name; their runtime requirements are declared as
# exec_depend in package.xml. The only compiled code here is the Pepper glue
# that runs on every scan or every odometry tick (src/).
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# The filters behind cloud_range_filter, free of ROS so the tests can drive
# them directly (test/test_cloud_filters.cpp).
add_library(cloud_filters STATIC
  src/cloud_filters.cpp
)
target_include_directories(cloud_filters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# cloud_range_filter: range / outlier / ground filters; feeds Nav2's
# /points_safety and /points_costmap and octomap's
# /cloud_registered_body_filtered.
add_executable(cloud_range_filter
  src/cloud_range_filter.cpp
)
target_link_libraries(cloud_range_filter cloud_filters)
ament_target_dependencies(cloud_range_filter
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
)

# wheel_odom_tf: odom -> base_footprint from /pepper_odom, for the loc stacks.
add_executable(wheel_odom_tf
  src/wheel_odom_tf.cpp
)
ament_target_dependencies(wheel_odom_tf
  rclcpp
  nav_msgs
  geometry_msgs
  tf2_ros
)

install(TARGETS cloud_range_filter wheel_odom_tf
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

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
  # Moved here from FAST_LIO/scripts. They are Pepper glue, not upstream
  # FAST-LIO code: point_lio, fastlio_lc_pgo, FAST-LIVO2 and
  # pepper_navigation all ran them out of fast_lio without declaring a
  # dependency on it.
  scripts/lio_odom_bridge.py
  # Imported by lio_odom_bridge.py rather than run directly. It must install
  # alongside it: Python puts the running script's own directory on sys.path,
  # and under --symlink-install that resolves to a directory holding both,
  # whether it lands on install/lib/pepper_slam or back on scripts/.
  scripts/lio_odom_guard.py
  scripts/pgo_map_odom_bridge.py
  scripts/cloud_to_odom_frame.py
  scripts/odom_compare.py
  scripts/check_frame_contract.py
  scripts/static_tf_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # headers are C++; left unset, cppcheck parses .h files as C
  set(ament_cmake_cppcheck_LANGUAGE "c++")
  # ROS 2 style; the .h headers are C++, which uncrustify must be told.
  set(ament_cmake_uncrustify_ADDITIONAL_ARGS LANGUAGE CPP)
  # flake8 is run manually below so rtabmap_base.launch.py (vendored
  # from an upstream rtabmap_ros example, not our style) can be excluded.
  # ament_cmake_flake8_FOUND doesn't suppress the auto-discovered hook here
  # (its extension gets registered before that variable takes effect) -
  # AMENT_LINT_AUTO_EXCLUDE is the mechanism ament_lint_auto actually
  # documents for "run this linter manually instead of automatically".
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_flake8 ament_cmake_pep257)
  # cpplint runs manually below, without its copyright rule: license
  # headers are kept out of the source files on purpose (see LICENSE).
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_cpplint)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_cpplint REQUIRED)
  ament_cpplint(FILTERS "-legal/copyright")

  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ament_flake8.ini")

  # pep257 is likewise run manually so it gets the same ignore list the
  # ament_python packages in this repo already use (see e.g.
  # text_to_speech/test/test_pep257.py). Codes disabled deliberately:
  #   D213  conflicts with D212, which the "ament" convention already ignores;
  #         the pair is mutually exclusive and this repo puts the summary on
  #         the first line.
  #   D205/D400/D415  assume a one-line summary; this repo wraps summaries
  #         across lines, so they fire on well-formed docstrings.
  #   D406/D407/D413  are numpy-style section rules; this repo uses
  #         Google-style "Args:"/"Returns:" sections.
  #   D401  imperative mood - too many false positives on descriptive
  #         module and test docstrings.
  find_package(ament_cmake_pep257 REQUIRED)
  ament_pep257(--add-ignore D205 D213 D400 D401 D406 D407 D413 D415)

  # Guards that *_map_odom_bridge levels the map by the same IMU the estimator
  # actually uses. Previously this file existed but was never registered, so it
  # only ever ran if someone invoked it by hand. It resolves sibling LIO
  # packages at runtime and skips (see the test) when they are not built --
  # they cannot be test_depends because fastlio_lc_pgo exec_depends on
  # pepper_slam, which would close a dependency cycle.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_leveling_frame_contract
    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_leveling_frame_contract.py")

  # The divergence guard is plain numpy with no ROS imports, so it tests
  # without a spin, a bag or a running estimator.
  ament_add_pytest_test(test_lio_odom_guard
    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_lio_odom_guard.py")

  # cloud_range_filter's range, outlier and ground-plane filters on synthetic
  # clouds, including a brute-force check of the outlier filter's grid.
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_cloud_filters test/test_cloud_filters.cpp)
  target_link_libraries(test_cloud_filters cloud_filters)
endif()

ament_package()
