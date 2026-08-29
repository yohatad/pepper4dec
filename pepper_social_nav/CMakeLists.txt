cmake_minimum_required(VERSION 3.8)
project(pepper_social_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_mppi_controller REQUIRED)
find_package(social_nav_msgs REQUIRED)
find_package(nav_2d_msgs REQUIRED)
find_package(dec_interfaces REQUIRED)

set(TRACKER_DEPS
  rclcpp geometry_msgs sensor_msgs std_msgs visualization_msgs
  tf2 tf2_ros tf2_geometry_msgs social_nav_msgs nav_2d_msgs dec_interfaces)

# --- people_tracker ---------------------------------------------------------
# Turns pixel detections into metric world-frame pedestrians with velocity.
# Everything else in this package is worthless without it.
add_library(people_tracker_core SHARED src/people_tracker.cpp)
target_include_directories(people_tracker_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(people_tracker_core Eigen3::Eigen)
ament_target_dependencies(people_tracker_core ${TRACKER_DEPS})

add_executable(people_tracker src/people_tracker_node.cpp)
target_link_libraries(people_tracker people_tracker_core)
ament_target_dependencies(people_tracker ${TRACKER_DEPS})

# --- costmap layer (pluginlib, loaded by nav2_costmap_2d) -------------------
add_library(pepper_social_nav_costmap_plugins SHARED src/proxemic_layer.cpp)
target_include_directories(pepper_social_nav_costmap_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(pepper_social_nav_costmap_plugins
  rclcpp pluginlib nav2_costmap_2d social_nav_msgs tf2_ros geometry_msgs)
pluginlib_export_plugin_description_file(nav2_costmap_2d costmap_plugins.xml)

# --- MPPI critic (pluginlib, loaded by nav2_mppi_controller) ----------------
add_library(pepper_social_nav_critics SHARED src/social_force_critic.cpp)
target_include_directories(pepper_social_nav_critics PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(pepper_social_nav_critics
  rclcpp pluginlib nav2_mppi_controller nav2_core nav2_costmap_2d
  social_nav_msgs tf2_geometry_msgs)
pluginlib_export_plugin_description_file(nav2_mppi_controller critics.xml)

install(TARGETS
  people_tracker_core
  pepper_social_nav_costmap_plugins
  pepper_social_nav_critics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS people_tracker RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py")
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml")
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING PATTERN "*.rviz")
install(PROGRAMS
  scripts/record_people_bag.sh
  scripts/social_metrics.py
  DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(people_tracker_core)
ament_export_dependencies(${TRACKER_DEPS} nav2_costmap_2d nav2_mppi_controller)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  # skip uncrustify: repo predates it and follows its own consistent style
  set(ament_cmake_uncrustify_FOUND TRUE)
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_flake8 ament_cmake_pep257)
  ament_lint_auto_find_test_dependencies()

  # flake8 / pep257 are run manually with the same ignore lists the rest of
  # the repo uses (see pepper_navigation/CMakeLists.txt for why each code is
  # disabled); AMENT_LINT_AUTO_EXCLUDE above is what stops the auto-discovered
  # hooks running them a second time with the default settings.
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ament_flake8.ini")
  find_package(ament_cmake_pep257 REQUIRED)
  ament_pep257(--add-ignore D205 D213 D400 D401 D406 D407 D413 D415)

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_people_tracker_geometry test/test_people_tracker_geometry.cpp)
  target_link_libraries(test_people_tracker_geometry people_tracker_core Eigen3::Eigen)
  ament_target_dependencies(test_people_tracker_geometry ${TRACKER_DEPS})
endif()

ament_package()
