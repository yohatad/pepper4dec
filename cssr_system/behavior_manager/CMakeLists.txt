cmake_minimum_required(VERSION 3.8)
project(behavior_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(tf2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_msgs REQUIRED) 
find_package(ament_cmake REQUIRED)
find_package(cssr_interfaces REQUIRED)
find_package(cssr_interfaces REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(behaviortree_ros2 REQUIRED)

include_directories(include)

add_executable(behavior_manager_node src/behavior_manager_implementation.cpp src/behavior_manager_application.cpp)

target_include_directories(behavior_manager_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(behavior_manager_node
  rclcpp
  behaviortree_cpp
  behaviortree_ros2
  nav2_msgs
  tf2
  tf2_geometry_msgs
  cssr_interfaces 
)

# target_link_libraries(behavior_manager_node
#   # ... other libs ...
#   tf2::tf2
#   rclcpp::rclcpp
#   behaviortree_cpp::behaviortree_cpp
#   behaviortree_ros2::behaviortree_ros2
#   tf2_geometry_msgs::tf2_geometry_msgs
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

# Install (so ros2 run works)
install(TARGETS behavior_manager_node
  DESTINATION lib/${PROJECT_NAME})
