cmake_minimum_required(VERSION 3.8)
project(overt_attention)

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
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(dec_interfaces REQUIRED)
find_package(naoqi_bridge_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Library
add_library(${PROJECT_NAME}_lib
  src/overt_attention_utilities.cpp
  src/overt_attention_bms.cpp
  src/overt_attention_implementation.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  std_msgs
  std_srvs
  geometry_msgs
  dec_interfaces
  naoqi_bridge_msgs
  cv_bridge
  ament_index_cpp
)
target_link_libraries(${PROJECT_NAME}_lib yaml-cpp ${OpenCV_LIBS})

# Executables — inherit all deps from ${PROJECT_NAME}_lib
add_executable(overt_attention_saliency src/overt_attention_saliency.cpp)
target_link_libraries(overt_attention_saliency ${PROJECT_NAME}_lib)

add_executable(overt_attention src/overt_attention_application.cpp)
target_link_libraries(overt_attention ${PROJECT_NAME}_lib)

add_executable(overt_attention_visualization src/overt_attention_visualization.cpp)
target_link_libraries(overt_attention_visualization ${PROJECT_NAME}_lib)

# Install
install(TARGETS
  overt_attention_saliency
  overt_attention
  overt_attention_visualization
  ${PROJECT_NAME}_lib
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
  sensor_msgs
  std_msgs
  std_srvs
  geometry_msgs
  dec_interfaces
  naoqi_bridge_msgs
  cv_bridge
  ament_index_cpp
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
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  # The helpers under test live in overt_attention_utilities.cpp; the inline
  # ones come from the interface header.
  ament_add_gtest(test_overt_attention_utilities
    test/test_overt_attention_utilities.cpp
  )
  target_link_libraries(test_overt_attention_utilities ${PROJECT_NAME}_lib)

  # BooleanMapSaliency lives in overt_attention_bms.cpp, split out of
  # overt_attention_saliency.cpp (which carries that node's main()) precisely
  # so it can be linked here.
  ament_add_gtest(test_boolean_map_saliency
    test/test_boolean_map_saliency.cpp
  )
  target_link_libraries(test_boolean_map_saliency ${PROJECT_NAME}_lib)
endif()

ament_package()
