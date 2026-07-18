cmake_minimum_required(VERSION 3.8)
project(gesture_execution)

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
find_package(dec_common REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(dec_interfaces REQUIRED)
find_package(naoqi_bridge_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)

# Library
add_library(${PROJECT_NAME}_lib
  src/gesture_execution_implementation.cpp
  src/pepper_kinematics_utilities.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  dec_interfaces
  naoqi_bridge_msgs
  ament_index_cpp
)
target_link_libraries(${PROJECT_NAME}_lib yaml-cpp)

# Executables — inherit all deps from ${PROJECT_NAME}_lib
add_executable(gesture_execution src/gesture_execution_application.cpp)
target_link_libraries(gesture_execution ${PROJECT_NAME}_lib dec_common::dec_common_runner)

add_executable(test_visualization src/gesture_test_visualization.cpp)
ament_target_dependencies(test_visualization rclcpp visualization_msgs geometry_msgs)

# Install
install(TARGETS
  gesture_execution
  test_visualization
  ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY data/    DESTINATION share/${PROJECT_NAME}/data)
install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)

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
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  # The kinematics helpers are pure math with no ROS dependencies, so the
  # test compiles the source directly instead of linking ${PROJECT_NAME}_lib.
  ament_add_gtest(test_pepper_kinematics
    test/test_pepper_kinematics.cpp
    src/pepper_kinematics_utilities.cpp
  )
  target_include_directories(test_pepper_kinematics PRIVATE include)
endif()

ament_package()
