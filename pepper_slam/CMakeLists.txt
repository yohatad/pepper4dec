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
  # Moved here from FAST_LIO/scripts. They are Pepper glue, not upstream
  # FAST-LIO code: point_lio, fastlio_lc_pgo, FAST-LIVO2 and
  # pepper_navigation all ran them out of fast_lio without declaring a
  # dependency on it.
  scripts/lio_odom_bridge.py
  scripts/pgo_map_odom_bridge.py
  scripts/cloud_range_filter.py
  scripts/cloud_to_odom_frame.py
  scripts/odom_compare.py
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
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_flake8 ament_cmake_pep257)
  ament_lint_auto_find_test_dependencies()

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
endif()

ament_package()
