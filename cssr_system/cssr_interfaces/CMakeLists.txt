cmake_minimum_required(VERSION 3.8)
project(cssr_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# Optional but fine to add:
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceDetection.msg"
  "msg/ObjectDetection.msg"
  "msg/OvertAttentionStatus.msg"
  "srv/OvertAttentionSetMode.srv"
  "srv/PerformGesture.srv"
  "srv/LanguageModelPrompt.srv"
  "srv/LanguageModelCreateCollection.srv"
  "action/Gesture.action"
  "action/Navigation.action"
  "action/SpeechRecognition.action"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
