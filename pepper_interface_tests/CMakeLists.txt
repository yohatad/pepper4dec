cmake_minimum_required(VERSION 3.5)
project(pepper_interface_tests)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Option to build for the simulator
option(SIMULATOR "Build for simulator without naoqi_driver and PEPPER_ROBOT definitions" OFF)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

if(NOT SIMULATOR)
  message(STATUS "Building for Pepper robot")
  find_package(naoqi_driver REQUIRED)
  add_definitions(-DPEPPER_ROBOT)
else()
  message(STATUS "Building for simulator")
endif()

# Include directories
include_directories(
  include
)

# Declare executables
add_executable(actuator_test
  src/actuatorTestImplementation.cpp
  src/actuatorTestApplication.cpp
)
target_include_directories(actuator_test PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${rclcpp_action_INCLUDE_DIRS}
)
target_link_libraries(actuator_test
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  control_msgs::control_msgs
  ${naoqi_driver_LIBRARIES}
)

add_executable(sensor_test
  src/sensorTestImplementation.cpp
  src/sensorTestApplication.cpp
)
target_include_directories(sensor_test PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sensor_test
  rclcpp::rclcpp
  cv_bridge::cv_bridge
  ${OpenCV_LIBS}
)

# Install targets
install(TARGETS
  actuator_test
  sensor_test
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  data
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
