cmake_minimum_required(VERSION 3.8)
project(cssr_system)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/FaceDetectionData.msg"
    "msg/OvertAttentionMode.msg"
    "srv/AnimateBehaviorSetActivation.srv"
    "srv/GestureExecutionPerformGesture.srv"
    "srv/OvertAttentionSetMode.srv"
    "srv/RobotLocalizationResetPose.srv"
    "srv/RobotNavigationSetGoal.srv"
    "srv/SpeechEventSetEnabled.srv"
    "srv/SpeechEventSetLanguage.srv"
    "srv/TabletEventPromptAndGetResponse.srv"
    "srv/TextToSpeechSayText.srv"
    DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

# Create executable
add_executable(behaviorController 
  src/behaviorControllerApplication.cpp
  src/behaviorControllerImplementation.cpp
  src/behaviorControllerUtilities.cpp
)

# Include directories
target_include_directories(behaviorController PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
ament_target_dependencies(behaviorController
  rclcpp
  std_msgs
  geometry_msgs
  behaviortree_cpp
  ament_index_cpp
  builtin_interfaces
)

# Link YAML-CPP
target_link_libraries(behaviorController ${YAML_CPP_LIBRARIES})
target_include_directories(behaviorController PRIVATE ${YAML_CPP_INCLUDE_DIRS})

# Link generated interfaces - Use the newer method
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(behaviorController "${cpp_typesupport_target}")

# Install executable
install(TARGETS behaviorController
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
  config
  data
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()