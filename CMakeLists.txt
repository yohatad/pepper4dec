cmake_minimum_required(VERSION 3.5)
project(cssr_system)

# --- C / C++ standards & warnings ----------------------------------------
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Find ament, ROS 2 & 3rd‑party deps ---------------------------------
find_package(ament_cmake                REQUIRED)
find_package(rclcpp                     REQUIRED)
find_package(std_msgs                   REQUIRED)
find_package(geometry_msgs              REQUIRED)
find_package(sensor_msgs                REQUIRED)
find_package(cv_bridge                  REQUIRED)
find_package(image_transport            REQUIRED)
find_package(rosidl_default_generators  REQUIRED)
find_package(behaviortree_cpp           QUIET)
find_package(yaml-cpp                   REQUIRED)

# --- Message/service generation -----------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  # face_detection messages
  "face_detection/msg/FaceDetectionMsgFile.msg"
  # behaviorController messages & services
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
ament_export_dependencies(rosidl_default_runtime)

# --- 1) Utilities library ------------------------------------------------
add_library(utilities
  utilities/src/cultureKnowledgeBaseImplementation.cpp
  utilities/src/environmentKnowledgeBaseImplementation.cpp
  utilities/src/behaviorControllerUtilities.cpp
)
target_include_directories(utilities
  PUBLIC
    # build‑time include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utilities/include>
    # install‑time include
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(utilities
  rclcpp
  std_msgs
  yaml-cpp
)

# --- 2) behaviorController executable ------------------------------------
if(behaviortree_cpp_FOUND)
  add_executable(behaviorController
    behaviorController/src/behaviorControllerImplementation.cpp
    behaviorController/src/behaviorControllerApplication.cpp
  )
  target_include_directories(behaviorController
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/behaviorController/include>
      $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(behaviorController
    rclcpp
    std_msgs
    geometry_msgs
    behaviortree_cpp
    yaml-cpp
  )
  target_link_libraries(behaviorController
    utilities
    yaml-cpp
  )
  rosidl_target_interfaces(
    behaviorController
    ${PROJECT_NAME}
    "rosidl_typesupport_cpp"
  )
  install(TARGETS behaviorController
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
else()
  message(WARNING "BehaviorTree.CPP not found; skipping behaviorController build.")
endif()

# --- 3) Install utilities library ---------------------------------------
install(TARGETS utilities
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# --- 4) Install all headers --------------------------------------------
# This will make #include "utilities/..." and #include "behaviorController/..."
# resolve correctly from the install space.
install(DIRECTORY
  utilities/include/
  behaviorController/include/
  DESTINATION include
)

# --- 5) Python node ------------------------------------------------------
install(PROGRAMS
  face_detection/src/face_detection_application.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- 6) face_detection assets -------------------------------------------
install(DIRECTORY
  face_detection/launch
  face_detection/models
  face_detection/config
  DESTINATION share/${PROJECT_NAME}/face_detection
)

# (Optional) if you ever add top‑level .launch.py files:
# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}/launch
# )

ament_package()
