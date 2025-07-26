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
find_package(rclcpp_action              REQUIRED)
find_package(std_msgs                   REQUIRED)
find_package(geometry_msgs              REQUIRED)
find_package(sensor_msgs                REQUIRED)
find_package(trajectory_msgs            REQUIRED)
find_package(control_msgs               REQUIRED)
find_package(cv_bridge                  REQUIRED)
find_package(image_transport            REQUIRED)
find_package(rosidl_default_generators  REQUIRED)
find_package(behaviortree_cpp           QUIET)
find_package(yaml-cpp                   REQUIRED)
find_package(OpenCV                     REQUIRED)
find_package(ament_index_cpp            REQUIRED)

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
  
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# --- 1) behaviorController executable ------------------------------------
if(behaviortree_cpp_FOUND)
  add_executable(behaviorController
    behaviorController/src/behaviorControllerImplementation.cpp
    behaviorController/src/behaviorControllerApplication.cpp
    behaviorController/src/behaviorControllerUtilities.cpp
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
    ament_index_cpp
  )
  target_link_libraries(behaviorController yaml-cpp)
  
  rosidl_target_interfaces(behaviorController ${PROJECT_NAME} "rosidl_typesupport_cpp")
  
  install(TARGETS behaviorController RUNTIME DESTINATION lib/${PROJECT_NAME})

else()
  message(WARNING "BehaviorTree.CPP not found; skipping behaviorController build.")
endif()

# # --- 3) overtAttention executable ----------------------------------------
# (omitted for brevity)

# --- 5) Install all headers --------------------------------------------
install(DIRECTORY
  behaviorController/include/
  DESTINATION include
)

# --- 6) Python node ------------------------------------------------------
install(PROGRAMS
  face_detection/src/face_detection_application.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- 7) Module assets ---------------------------------------------------
install(DIRECTORY
  face_detection/launch
  face_detection/models
  face_detection/config
  DESTINATION share/${PROJECT_NAME}/face_detection
)

# Install behaviorController config where the code expects it
install(DIRECTORY
  behaviorController/config/
  DESTINATION share/${PROJECT_NAME}/behaviorController/config
)

# Keep behaviorController’s data grouped under its own folder
install(DIRECTORY
  behaviorController/data/
  DESTINATION share/${PROJECT_NAME}/behaviorController/data
)

# (Optional) other installs…

ament_package()
