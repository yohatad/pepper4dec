cmake_minimum_required(VERSION 3.5)
project(cssr_system)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find build tools and dependencies
find_package(ament_cmake                REQUIRED)
find_package(rclcpp                     REQUIRED)
find_package(std_msgs                   REQUIRED)
find_package(geometry_msgs              REQUIRED)
find_package(sensor_msgs                REQUIRED)
find_package(cv_bridge                  REQUIRED)
find_package(image_transport            REQUIRED)
find_package(rosidl_default_generators  REQUIRED)

# BehaviorTree.CPP dependency (if available)
find_package(behaviortree_cpp QUIET)

# --- Message generation --------------------------------------------------

rosidl_generate_interfaces(${PROJECT_NAME}
  "face_detection/msg/FaceDetection.msg"
  "behaviorController/msg/FaceDetectionData.msg"
  "behaviorController/msg/OvertAttentionMode.msg"
  "behaviorController/srv/AnimateBehaviorSetActivation.srv"
  "behaviorController/srv/GestureExecutionPerformGesture.srv"
  "behaviorController/srv/OvertAttentionSetMode.srv"
  "behaviorController/srv/RobotLocalizationResetPose.srv"
  "behaviorController/srv/RobotNavigationSetGoal.srv"
  "behaviorController/srv/SpeechEventSetEnabled.srv"
  "behaviorController/srv/SpeechEventSetLanguage.srv"
  "behaviorController/srv/TabletEventPromptAndGetResponse.srv"
  "behaviorController/srv/TextToSpeechSayText.srv"
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

# --- C++ executables ------------------------------------------------------

# behaviorController executable (if BehaviorTree.CPP is available)
if(behaviortree_cpp_FOUND)
  # Utilities library
  add_library(utilities 
    behaviorController/include/utilities/src/cultureKnowledgeBaseImplementation.cpp
    behaviorController/include/utilities/src/environmentKnowledgeBaseImplementation.cpp
    behaviorController/include/utilities/src/behaviorControllerUtilities.cpp
  )
  target_include_directories(utilities PRIVATE 
    behaviorController/include/utilities/include
    behaviorController/include
  )
  ament_target_dependencies(utilities rclcpp std_msgs)

  # behaviorController executable
  add_executable(behaviorController 
    behaviorController/src/behaviorControllerImplementation.cpp 
    behaviorController/src/behaviorControllerApplication.cpp
  )
  target_include_directories(behaviorController PRIVATE 
    behaviorController/include
    ${CMAKE_CURRENT_SOURCE_DIR}/behaviorController/include
  )
  target_link_libraries(behaviorController utilities)
  ament_target_dependencies(behaviorController 
    rclcpp 
    std_msgs 
    geometry_msgs
    behaviortree_cpp
  )
  rosidl_target_interfaces(behaviorController ${PROJECT_NAME} "rosidl_typesupport_cpp")
  
  install(TARGETS behaviorController utilities
    DESTINATION lib/${PROJECT_NAME}
  )
  
  # Install behaviorController assets
  install(
    DIRECTORY
      behaviorController/config
      behaviorController/data
    DESTINATION share/${PROJECT_NAME}/behaviorController
  )
else()
  message(WARNING "BehaviorTree.CPP not found. behaviorController will not be built.")
endif()

# --- Python nodes ---------------------------------------------------------

install(PROGRAMS
  face_detection/src/face_detection_application.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- Install launch files & assets ---------------------------------------

# # Main launch directory
# install(
#   DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}/launch
# )

# Per-node launch/config/models directories
install(
  DIRECTORY
    face_detection/launch
    face_detection/models
    face_detection/config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
