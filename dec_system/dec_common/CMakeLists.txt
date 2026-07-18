cmake_minimum_required(VERSION 3.8)
project(dec_common)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# Compiled library: the shared ByteTracker, the CameraLifecycleNode base for
# the perception nodes, and the param_loader.h header-only utilities.
add_library(${PROJECT_NAME}
  src/byte_tracker.cpp
  src/camera_lifecycle_node.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp_lifecycle sensor_msgs message_filters cv_bridge ament_index_cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen yaml-cpp ${OpenCV_LIBS})

install(DIRECTORY include/ DESTINATION include/)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp_lifecycle Eigen3 sensor_msgs message_filters
  cv_bridge ament_index_cpp yaml-cpp OpenCV)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  # skip uncrustify: repo predates it and follows its own consistent style
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_byte_tracker test/test_byte_tracker.cpp)
  target_link_libraries(test_byte_tracker ${PROJECT_NAME})
endif()

ament_package()
