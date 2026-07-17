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
find_package(dec_interfaces REQUIRED)
find_package(naoqi_bridge_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Library
add_library(${PROJECT_NAME}_lib
  src/overt_attention_utilities.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(${PROJECT_NAME}_lib
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
target_link_libraries(${PROJECT_NAME}_lib yaml-cpp ${OpenCV_LIBS})

# Executables — inherit all deps from ${PROJECT_NAME}_lib
add_executable(overt_attention_saliency src/overt_attention_saliency.cpp)
target_link_libraries(overt_attention_saliency ${PROJECT_NAME}_lib)

add_executable(overt_attention_unified_attention src/overt_attention_unified_attention.cpp)
target_link_libraries(overt_attention_unified_attention ${PROJECT_NAME}_lib)

add_executable(overt_attention_visualization src/overt_attention_visualization.cpp)
target_link_libraries(overt_attention_visualization ${PROJECT_NAME}_lib)

# Install
install(TARGETS
  overt_attention_saliency
  overt_attention_unified_attention
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

ament_package()
