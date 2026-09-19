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

# Note: generate_keepout.py moved to ros2_ws/utils/ -- it is an offline
# map-editing utility (no rclpy), so it does not belong in a ROS package.
# Run it manually: python3 utils/generate_keepout.py

# Install python helper scripts
install(PROGRAMS
  scripts/wait_for_map_then_start.py
  scripts/localization_recovery.py
  scripts/localization_watchdog.py
  scripts/tf_nav_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files.
#
# NOT symlinked, unlike config/ and rviz/ below: `ros2 launch <pkg> <file>`
# resolves its top-level file argument via ros2launch/api/api.py's
# os.walk(package_share_directory) with the default followlinks=False, which
# never descends into a SYMLINKED launch/ directory -- MEASURED: symlinking
# this one broke `ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py`
# outright ("file ... was not found"), even though the file plainly existed
# through the symlink. config/ and rviz/ are safe because they're only ever
# reached by direct path construction inside a launch file's own Python code
# (get_package_share_directory() + os.path.join()), never through that
# top-level os.walk lookup.
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

# Install rviz configs. SYMLINKED -- same reasoning as config/ above.
install(CODE "set(_rviz_src \"${CMAKE_CURRENT_SOURCE_DIR}/rviz\")")
install(CODE "set(_rviz_pkg \"${PROJECT_NAME}\")")
install(CODE [[
  set(_rviz_dst "${CMAKE_INSTALL_PREFIX}/share/${_rviz_pkg}/rviz")
  get_filename_component(_rviz_parent "${_rviz_dst}" DIRECTORY)
  file(MAKE_DIRECTORY "${_rviz_parent}")
  if(EXISTS "${_rviz_dst}" OR IS_SYMLINK "${_rviz_dst}")
    file(REMOVE_RECURSE "${_rviz_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_rviz_src}" "${_rviz_dst}"
    RESULT_VARIABLE _rviz_rc ERROR_QUIET)
  if(_rviz_rc EQUAL 0)
    message(STATUS "rviz/ symlinked -> ${_rviz_src} (edits are live)")
  else()
    message(STATUS "rviz/ symlink failed -- copying instead")
    file(COPY "${_rviz_src}/" DESTINATION "${_rviz_dst}"
      FILES_MATCHING PATTERN "*.rviz")
  endif()
]])

# Install config files.
#
# SYMLINKED, not copied, so tuning is live: nav2 reads
# share/<pkg>/config/<file>.yaml, and with a symlink that IS the source file.
# Edit it -- by hand or from nav2_config's Save Config -- relaunch, done. No
# colcon build in the loop.
#
# install(DIRECTORY) copies unconditionally, and colcon's --symlink-install does
# NOT change that: it applies to Python packages and a few install types, not to
# install(DIRECTORY). Without this, every save left nav2 reading the OLD values
# with nothing to indicate why -- the config looked edited and had no effect.
#
# Falls back to a copy if the symlink cannot be created, so the package still
# installs correctly and only the live-editing convenience is lost.
#
# Two install(CODE) blocks: the first carries the configure-time paths in, the
# second is a bracket argument so the script needs no backslash escaping.
# NOTE: no ';' anywhere inside these blocks -- CMake treats it as a list
# separator and silently truncates the generated script mid-statement.
install(CODE "set(_cfg_src \"${CMAKE_CURRENT_SOURCE_DIR}/config\")")
install(CODE "set(_cfg_pkg \"${PROJECT_NAME}\")")
install(CODE [[
  set(_cfg_dst "${CMAKE_INSTALL_PREFIX}/share/${_cfg_pkg}/config")
  get_filename_component(_cfg_parent "${_cfg_dst}" DIRECTORY)
  file(MAKE_DIRECTORY "${_cfg_parent}")
  if(EXISTS "${_cfg_dst}" OR IS_SYMLINK "${_cfg_dst}")
    file(REMOVE_RECURSE "${_cfg_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_cfg_src}" "${_cfg_dst}"
    RESULT_VARIABLE _cfg_rc ERROR_QUIET)
  if(_cfg_rc EQUAL 0)
    message(STATUS "config/ symlinked -> ${_cfg_src} (edits are live)")
  else()
    message(STATUS "config/ symlink failed -- copying instead")
    file(COPY "${_cfg_src}/" DESTINATION "${_cfg_dst}"
      FILES_MATCHING PATTERN "*.yaml")
  endif()
]])

# Install 2D occupancy grid files (nav2 map_server format).
install(DIRECTORY
  map/
  DESTINATION share/${PROJECT_NAME}/map
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.pgm"
)

# Install the 3D map fastlio_lc_pgo's /pgo_batch_optimize writes
# (map_pcd_path). It shares the levelled, floor-referenced frame with the
# .pgm/.yaml 2D grid in map/, but lives in its own directory since it is a
# different artifact (3D cloud vs. 2D nav grid). Nothing localizes against it
# directly any more -- fastlio_localization reads the per-keyframe pose.json
# and clouds below -- but it is still the PGO run's map product.
install(DIRECTORY
  pcd/
  DESTINATION share/${PROJECT_NAME}/pcd
  FILES_MATCHING
  PATTERN "*.pcd"
  # Levelled keyframe poses in KITTI 3x4, written beside the .pcd by the same
  # PGO run. No longer a runtime input (that was lio_localization's global
  # search); kept as the run's record, and as the plain-text counterpart to
  # the pose.json below.
  PATTERN "*_poses.txt"
  # pose.json is the SAME mapping run in the format fastlio_localization reads:
  # per-keyframe quaternion poses, against the KITTI 3x4 in *_poses.txt. It
  # ships here so every artifact of the run stays visibly ONE set rather
  # than files that happen to describe the same building. The 2735 keyframe
  # CLOUDS it indexes are 75 MB and deliberately do NOT live in this package --
  # they are in ros2_ws/bags/pcd, addressed by localization.map_scan_dir.
  PATTERN "*.json"
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
  # skip uncrustify: repo predates it and follows its own consistent style
  set(ament_cmake_uncrustify_FOUND TRUE)
  # flake8 is run manually below so it picks up ament_flake8.ini (the shared
  # extend-ignore list). ament_cmake_flake8_FOUND doesn't suppress the
  # auto-discovered hook here (its extension gets registered before that
  # variable takes effect) - AMENT_LINT_AUTO_EXCLUDE is the mechanism
  # ament_lint_auto actually documents for "run this linter manually".
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

  # Guards that the 3 nav2 sections meant to be shared have not drifted apart
  # across the 4 param files. Previously this file existed but was never
  # registered, so it only ever ran if someone invoked it by hand.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_shared_nav2_params
    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_shared_nav2_params.py")
endif()

ament_package()
