cmake_minimum_required(VERSION 3.8)
project(person_detection)

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
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(dec_interfaces REQUIRED)
find_package(dec_common REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(onnxruntime_vendor REQUIRED)

# Library
add_library(${PROJECT_NAME}_lib
  src/person_detection_implementation.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  geometry_msgs
  sensor_msgs
  dec_interfaces
  dec_common
  message_filters
  cv_bridge
  ament_index_cpp
  onnxruntime_vendor
)
target_link_libraries(${PROJECT_NAME}_lib yaml-cpp ${OpenCV_LIBS})

# Executable
add_executable(person_detection src/person_detection_application.cpp)
target_link_libraries(person_detection ${PROJECT_NAME}_lib)

# Install
install(TARGETS
  person_detection
  ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/  DESTINATION include/)
install(DIRECTORY config/   DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY data/     DESTINATION share/${PROJECT_NAME}/data)
# models/ is git-ignored (large ONNX weights, not checked in) and so is
# absent on a fresh clone (e.g. CI) until manually downloaded - guard the
# install so that absence doesn't hard-fail the build.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
  install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
endif()
install(DIRECTORY launch/   DESTINATION share/${PROJECT_NAME}/launch)

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

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_bag_replay_launch.py
    TIMEOUT 300
  )
endif()

ament_package()
