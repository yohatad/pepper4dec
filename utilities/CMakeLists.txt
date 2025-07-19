cmake_minimum_required(VERSION 3.8)
project(utilities)

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
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(include)

# Create library for utilities
add_library(${PROJECT_NAME}
  src/behaviorControllerUtilities.cpp
  src/cultureKnowledgeBaseImplementation.cpp
  src/environmentKnowledgeBaseImplementation.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  yaml-cpp
)

# Create executables for testing/examples
add_executable(culture_knowledge_base_example
  src/cultureKnowledgeBaseApplication.cpp
)

target_link_libraries(culture_knowledge_base_example
  ${PROJECT_NAME}
)

ament_target_dependencies(culture_knowledge_base_example
  rclcpp
  yaml-cpp
)

add_executable(environment_knowledge_base_example
  src/environmentKnowledgeBaseApplication.cpp
)

target_link_libraries(environment_knowledge_base_example
  ${PROJECT_NAME}
)

ament_target_dependencies(environment_knowledge_base_example
  rclcpp
  yaml-cpp
)

# Install targets
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  culture_knowledge_base_example
  environment_knowledge_base_example
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

# Install config and data files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config/
)

install(DIRECTORY
  data/
  DESTINATION share/${PROJECT_NAME}/data/
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Export dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp yaml-cpp)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
