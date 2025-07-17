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

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- Message generation --------------------------------------------------

rosidl_generate_interfaces(${PROJECT_NAME}
  "face_detection/msg/face_detection_msg_file.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# --- C++ nodes ----------------------------------------------------------

# gestureExecution_node
add_executable(gesture_execution_node
  gestureExecution/src/gesture_execution_node.cpp
)
target_include_directories(gesture_execution_node
  PRIVATE gestureExecution/include
)
ament_target_dependencies(gesture_execution_node
  rclcpp std_msgs
)
install(TARGETS gesture_execution_node
  DESTINATION lib/${PROJECT_NAME}
)

# (add more C++ executables here if needed)

# --- Python nodes -------------------------------------------------------

install(PROGRAMS
  animateBehaviour/src/animate_node.py
  behaviorController/src/behavior_controller_node.py
  face_detection/src/face_detection_node.py
  overtAttention/src/overt_attention_node.py
  person_detection/src/person_detection_node.py
  sound_detection/src/sound_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- Install assets & launch files ------------------------------------

# umbrella launch
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

# each node’s own launch & assets
install(
  DIRECTORY
    face_detection/launch
    gestureExecution/launch
    animateBehaviour/models
    face_detection/models
    person_detection/models
    animateBehaviour/config
    behaviorController/config
    face_detection/config
    overtAttention/config
    sound_detection/config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
