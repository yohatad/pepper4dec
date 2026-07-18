+# =============================================================================
# Pepper4DEC Docker Image
# Digital Experience Center - Autonomous Pepper Robot Tour System
# =============================================================================
#
# This Dockerfile builds a complete ROS2 Humble environment with all packages
# needed for the Pepper robot tour system at the Upanzi Digital Experience Center.
#
# Usage:
#   docker build -t pepper4dec:latest .
#   docker run -it --privileged pepper4dec:latest
#
# For GPU support (NVIDIA):
#   docker build -t pepper4dec:latest --build-arg USE_CUDA=1 .
#   docker run -it --gpus all --privileged pepper4dec:latest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Base Image - ROS2 Humble
# -----------------------------------------------------------------------------
FROM ros:humble-perceptive-jammy AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# -----------------------------------------------------------------------------
# Build Arguments
# -----------------------------------------------------------------------------
ARG USE_CUDA=0
ARG PYTHON_VERSION=3.10

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    \
    # ROS2 build tools
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-pip \
    \
    # ROS2 Humble packages
    ros-humble-nav2-bringup \
    ros-humble-nav2-map-server \
    ros-humble-nav2-amcl \
    ros-humble-nav2-controller \
    ros-humble-nav2-planner \
    ros-humble-nav2-behaviors \
    ros-humble-nav2-bt-navigator \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-nav2-costmap-2d \
    ros-humble-slam-toolbox \
    ros-humble-rtabmap-ros \
    ros-humble-realsense2-camera \
    ros-humble-yolo-v8 \
    ros-humble-message-filters \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-nav2-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-std-srvs \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-naoqi-libqi \
    ros-humble-naoqi-libqicore \
    ros-humble-pepper-meshes \
    ros-humble-nao-meshes \
    \
    # Additional libraries
    libyaml-cpp-dev \
    libopencv-dev \
    libomp-dev \
    libeigen3-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libudev-dev \
    libusb-1.0-0 \
    \
    # Python development
    python3-dev \
    python3-venv \
    python3-pip \
    python3-setuptools \
    \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Virtual Environment Setup
# -----------------------------------------------------------------------------
RUN python3 -m venv /opt/pepper4dec-venv
ENV VIRTUAL_ENV=/opt/pepper4dec-venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Install PyTorch (Conditional - for GPU support)
# -----------------------------------------------------------------------------
# Note: PyTorch installation depends on CUDA version
# Uncomment the appropriate version for your GPU

# For CUDA 12.1 (default)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# For CPU only (uncomment if not using GPU)
# RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------------------------
RUN pip install \
    # Core ML/AI dependencies
    numpy==1.26.3 \
    scipy==1.14.1 \
    opencv-python==4.10.0.84 \
    onnxruntime-gpu==1.19.0 \
    onnx \
    pillow \
    scikit-learn \
    \
    # Sentence transformers for embeddings
    sentence-transformers \
    \
    # ChromaDB for vector database
    chromadb \
    \
    # OpenAI client (for LLM integration)
    openai \
    \
    # YAML parsing
    pyyaml \
    \
    # Additional utilities
    lap==0.5.12 \
    sounddevice \
    soundfile \
    kokoro \
    \
    # ROS2 Python dependencies
    rclpy \
    std_msgs \
    geometry_msgs \
    sensor_msgs \
    nav_msgs \
    action_msgs \
    \
    # Cleanup
    && pip cache purge

# -----------------------------------------------------------------------------
# Create workspace and clone repositories
# -----------------------------------------------------------------------------
RUN mkdir -p /home/pepper/ros2_ws/src
WORKDIR /home/pepper/ros2_ws

# Clone NAOqi driver repositories
# User's fork for naoqi_bridge_msgs
RUN git clone https://github.com/yohatad/naoqi_driver_bridge_msgs2.git src/naoqi_bridge_msgs

# Clone naoqi_driver2 (user's fork)
RUN git clone https://github.com/yohatad/naoqi_driver2.git src/naoqi_driver2

# Clone other NAOqi dependencies from ros-naoqi
RUN git clone https://github.com/ros-naoqi/libqi.git -b ros2 src/naoqi_libqi && \
    git clone https://github.com/ros-naoqi/libqicore.git -b ros2 src/naoqi_libqicore && \
    git clone https://github.com/ros-naoqi/nao_meshes2.git src/nao_meshes && \
    git clone https://github.com/ros-naoqi/pepper_meshes2.git src/pepper_meshes

# Clone BehaviorTree.CPP and BehaviorTree.ROS2
RUN git clone https://github.com/BehaviorTree/BehaviorTree.CPP.git src/BehaviorTree.CPP && \
    git clone https://github.com/BehaviorTree/BehaviorTree.ROS2.git src/BehaviorTree.ROS2

# Clone the main repository
RUN git clone https://github.com/yohatad/pepper4dec.git src/pepper4dec

# -----------------------------------------------------------------------------
# Install ROS2 package dependencies
# -----------------------------------------------------------------------------
RUN cd /home/pepper/ros2_ws && \
    rosdep install --from-paths src --ignore-src -r -y

# -----------------------------------------------------------------------------
# Build ROS2 packages
# -----------------------------------------------------------------------------
RUN cd /home/pepper/ros2_ws && \
    I_AGREE_TO_NAO_MESHES_LICENSE=1 I_AGREE_TO_PEPPER_MESHES_LICENSE=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# -----------------------------------------------------------------------------
# Install package-specific Python requirements
# -----------------------------------------------------------------------------
RUN pip install --no-deps -r /home/pepper/ros2_ws/src/pepper4dec/dec_system/face_detection/requirements.txt && \
    pip install --no-deps -r /home/pepper/ros2_ws/src/pepper4dec/dec_system/conversation_manager/requirements.txt && \
    pip install --no-deps -r /home/pepper/ros2_ws/src/pepper4dec/dec_system/text_to_speech/requirements.txt

# -----------------------------------------------------------------------------
# Download Model Files (Optional - for face detection)
# -----------------------------------------------------------------------------
# Note: Model files should be placed in the respective models/ directories
# - face_detection/models/face_detection_goldYOLO.onnx
# - face_detection/models/face_detection_sixdrepnet360.onnx
# - person_detection/models/yolov8n.onnx

# -----------------------------------------------------------------------------
# Environment Setup Script
# -----------------------------------------------------------------------------
RUN echo '#!/bin/bash' > /opt/pepper4dec_setup.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /opt/pepper4dec_setup.sh && \
    echo 'source /home/pepper/ros2_ws/install/setup.bash' >> /opt/pepper4dec_setup.sh && \
    echo 'source /opt/pepper4dec-venv/bin/activate' >> /opt/pepper4dec_setup.sh && \
    echo 'export LANG=C.UTF-8' >> /opt/pepper4dec_setup.sh && \
    echo 'export LC_ALL=C.UTF-8' >> /opt/pepper4dec_setup.sh && \
    echo 'export ROS_DOMAIN_ID=42' >> /opt/pepper4dec_setup.sh && \
    chmod +x /opt/pepper4dec_setup.sh

# -----------------------------------------------------------------------------
# Create non-root user (optional)
# -----------------------------------------------------------------------------
RUN useradd -m -s /bin/bash pepper && \
    chown -R pepper:pepper /home/pepper

# -----------------------------------------------------------------------------
# Final Configuration
# -----------------------------------------------------------------------------
WORKDIR /home/pepper/ros2_ws

# Set default environment
ENV BASH_ENV=/opt/pepper4dec_setup.sh
ENV ROS_ROOT=/home/pepper/ros2_ws
ENV ROS_PACKAGE_PATH=/home/pepper/ros2_ws/src/pepper4dec/dec_system:/opt/ros/humble/share

# Default command
CMD ["/bin/bash"]