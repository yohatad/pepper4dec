cmake_minimum_required(VERSION 3.8)
project(pepper_navigation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
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
find_package(action_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(naoqi_bridge_msgs REQUIRED)

# pepper_navigation_lib (src/lib/): the decisions the nodes make, free of ROS
# so the tests drive them with time passed in. bumper_latch.h and
# topic_liveness.h are header-only.
#   localization_health.cpp  localization_watchdog's lost/cancel logic
#                            (test_localization_health)
add_library(${PROJECT_NAME}_lib STATIC
  src/lib/localization_health.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set_target_properties(${PROJECT_NAME}_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The runtime nodes, one executable each (src/<node>.cpp):
#   bumper_guard             last gate before cmd_vel; a bumper contact blocks
#                            its direction and leaves reverse open for BackUp
#   tf_nav_relay             /tf without Pepper's joint tree, on /tf_nav
#                            (~4150 transforms/s, so C++)
#   localization_watchdog    cancels goals while the localizer reports lost
#   localization_recovery    /localization_recover in front of every backend
#   wait_for_map_then_start  starts Nav2 once map -> base_footprint exists
set(NODES
  bumper_guard
  tf_nav_relay
  localization_watchdog
  localization_recovery
  wait_for_map_then_start
)
foreach(node ${NODES})
  add_executable(${node} src/${node}.cpp)
  target_link_libraries(${node} ${PROJECT_NAME}_lib)
endforeach()
ament_target_dependencies(bumper_guard
  rclcpp geometry_msgs sensor_msgs diagnostic_msgs std_srvs naoqi_bridge_msgs)
ament_target_dependencies(tf_nav_relay rclcpp tf2_msgs)
ament_target_dependencies(localization_watchdog
  rclcpp action_msgs diagnostic_msgs std_srvs)
ament_target_dependencies(localization_recovery rclcpp std_srvs)
ament_target_dependencies(wait_for_map_then_start rclcpp nav2_msgs tf2 tf2_ros)
install(TARGETS ${NODES}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Dev/debug tooling (src/tools/), not part of the navigation pipeline.
#   odom_path_publisher  /pepper_odom as a nav_msgs/Path + RViz markers, for
#                        closed-loop odometry testing (launch/tools/odom_test)
#   send_goal            one-shot Nav2 goal sender (the relevant subset of
#                        nav2_simple_commander's Python-only BasicNavigator)
add_executable(odom_path_publisher src/tools/odom_path_publisher.cpp)
ament_target_dependencies(odom_path_publisher
  rclcpp nav_msgs geometry_msgs visualization_msgs std_srvs)
add_executable(send_goal src/tools/send_goal.cpp)
ament_target_dependencies(send_goal rclcpp rclcpp_action geometry_msgs nav2_msgs)
install(TARGETS odom_path_publisher send_goal
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Python tools, installed as executables. Not part of the navigation pipeline.
install(PROGRAMS
  # bag_test only: replay helpers (see "Bag replay" in README.md)
  scripts/bag_test/cloud_delay.py
  scripts/bag_test/costmap_explain.py
  scripts/bag_test/depth_to_cloud.py
  scripts/bag_test/points_to_image.py
  scripts/bag_test/cm_bag_judge.py
  # live tuning aid: colours what the safety chain cuts (safety_cut_viz.launch.py)
  scripts/safety_cut_viz.py
  DESTINATION lib/${PROJECT_NAME}
)

# pepper_navigation/nav2_nodes.py: the nodes every Nav2 profile starts, which
# the launch files import.
ament_python_install_package(${PROJECT_NAME})

# Install launch, config, rviz, the 2D map and the prior map: one plain
# install(DIRECTORY), matching pepper_slam and FAST_LIO.
#
# With colcon --symlink-install this installs a REAL directory per folder with
# one SYMLINK PER FILE back into src/, so edits are live without a rebuild
# (a NEW file still needs one).
#
# Do NOT add FILES_MATCHING / PATTERN filters: --symlink-install cannot
# symlink through them and silently falls back to COPYING (see
# pepper_slam/CMakeLists.txt). That fallback is why this package once carried
# hand-written install(CODE) symlink blocks. And do NOT symlink a whole
# directory instead: `ros2 launch <pkg> <file>` finds its file with
# os.walk(followlinks=False), which skips a symlinked launch/ directory.
install(DIRECTORY
  launch config rviz map pcd
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # headers are C++; left unset, cppcheck parses .h files as C
  set(ament_cmake_cppcheck_LANGUAGE "c++")
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # flake8 is run manually below so it picks up ament_flake8.ini (the shared
  # extend-ignore list). ament_cmake_flake8_FOUND doesn't suppress the
  # auto-discovered hook here (its extension gets registered before that
  # variable takes effect) - AMENT_LINT_AUTO_EXCLUDE is the mechanism
  # ament_lint_auto actually documents for "run this linter manually".
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_flake8 ament_cmake_pep257)
  # ROS 2 style; the .h headers are C++, which uncrustify must be told.
  set(ament_cmake_uncrustify_ADDITIONAL_ARGS LANGUAGE CPP)
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

  find_package(ament_cmake_gtest REQUIRED)
  # bumper_guard's latch/direction logic, with time passed in so the cases
  # are exact instead of slept through.
  ament_add_gtest(test_bumper_latch test/test_bumper_latch.cpp)
  target_include_directories(test_bumper_latch PRIVATE include)

  # The chain's only fail-safe on sensor loss: Humble's collision_monitor
  # holds its last verdict when a source goes stale (MEASURED, see the header).
  ament_add_gtest(test_topic_liveness test/test_topic_liveness.cpp)
  target_include_directories(test_topic_liveness PRIVATE include)

  # localization_watchdog's decisions on a simulated clock: lost_duration,
  # stale_timeout, the recancel cadence and the release.
  ament_add_gtest(test_localization_health test/test_localization_health.cpp)
  target_link_libraries(test_localization_health ${PROJECT_NAME}_lib)

  # Guards that the 3 nav2 sections meant to be shared have not drifted apart
  # across the nav2 param files.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_shared_nav2_params
    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_shared_nav2_params.py")

  # The real collision monitor on the benchmark bag, scored scan by scan
  # against polygon geometry by cm_bag_judge. ~90 s; skips without the bag.
  ament_add_pytest_test(test_collision_monitor_bag
    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_collision_monitor_bag.py"
    TIMEOUT 300)
endif()

ament_package()
