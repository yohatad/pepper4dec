cmake_minimum_required(VERSION 3.8)
project(behavior_controller)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(dec_interfaces REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(naoqi_bridge_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Library
add_library(${PROJECT_NAME}_lib
  src/behaviorControllerImplementation.cpp
  src/behaviorControllerUtilities.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_action
  std_msgs
  std_srvs
  geometry_msgs
  nav2_msgs
  naoqi_bridge_msgs
  dec_interfaces
  behaviortree_cpp
  behaviortree_ros2
  ament_index_cpp
)
target_link_libraries(${PROJECT_NAME}_lib yaml-cpp)

# Executable — inherits all deps from ${PROJECT_NAME}_lib
add_executable(behaviorController src/behaviorControllerApplication.cpp)
target_link_libraries(behaviorController ${PROJECT_NAME}_lib)

# Install
install(TARGETS behaviorController ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/  DESTINATION include/)
install(DIRECTORY config/   DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY data/     DESTINATION share/${PROJECT_NAME}/data)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
endif()

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_export_dependencies(
  rclcpp
  rclcpp_action
  std_msgs
  std_srvs
  geometry_msgs
  nav2_msgs
  naoqi_bridge_msgs
  dec_interfaces
  behaviortree_cpp
  behaviortree_ros2
  ament_index_cpp
)

ament_package()