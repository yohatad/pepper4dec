cmake_minimum_required(VERSION 3.8)
project(dec_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# Optional but fine to add:
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceDetection.msg"
  "msg/PersonDetection.msg"
  "srv/ConversationManagerPrompt.srv"
  "srv/GetDepthROI.srv"
  "action/ConversationManager.action"
  "action/TTS.action"
  "action/Gesture.action"
  "action/AnimateBehavior.action"
  "action/SpeechRecognition.action"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # ROS 2 style; see the C++ packages for why these two are set.
  set(ament_cmake_uncrustify_ADDITIONAL_ARGS LANGUAGE CPP)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_cpplint)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_cpplint REQUIRED)
  ament_cpplint(FILTERS "-legal/copyright")

  # Round-trip every generated type through the CDR serializer.
  find_package(ament_cmake_gtest REQUIRED)
  find_package(rclcpp REQUIRED)
  ament_add_gtest(test_serialization test/test_serialization.cpp)
  rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
  target_link_libraries(test_serialization "${cpp_typesupport_target}" rclcpp::rclcpp)
endif()

ament_package()
