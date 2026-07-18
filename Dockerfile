# syntax=docker/dockerfile:1.7
# =============================================================================
# Pepper4DEC Docker Image
# Digital Experience Center - Autonomous Pepper Robot Tour System
# =============================================================================
#
# The dec_system nodes CANNOT share one Python environment: speech_event needs
# numpy 2.2.6 while conversation_manager needs numpy 1.24.4, and each pulls a
# different torch/CUDA build. Each node therefore gets its own venv, built in
# its own stage from a lockfile, and they are assembled into the final image.
#
# The venvs land at $HOME/ros2_ws/.venvs -- the same layout as the host -- so
# the dec_system/*/scripts/ launchers, which source
# "$HOME/ros2_ws/.venvs/venv_map.sh", work unchanged in both places.
#
# Usage:
#   docker compose build && docker compose up
#   docker build -t pepper4dec:latest .            # full image
#   docker build --target venv-sound -t x .        # single-venv slim image
# =============================================================================

ARG ROS_DISTRO=humble
ARG PYTHON_VERSION=3.10
ARG WS=/home/pepper/ros2_ws

# -----------------------------------------------------------------------------
# base - system deps shared by every stage
# -----------------------------------------------------------------------------
FROM ros:humble-perception-jammy AS base

ARG WS
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # The torch/CUDA wheels are 300-800MB each and the connection drops
    # mid-stream often enough to fail a build. --retries only covers
    # establishing a connection, so it does NOT help with a truncated
    # response (IncompleteRead); --resume-retries is what resumes a partial
    # download. Set as env vars so all three venv stages inherit them;
    # the venv's bundled older pip ignores names it does not recognise.
    PIP_RESUME_RETRIES=5 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=60

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget curl vim tmux htop \
      python3-colcon-common-extensions python3-vcstool python3-rosdep python3-pip \
      python3-dev python3-venv python3-setuptools \
      ros-humble-nav2-bringup ros-humble-nav2-map-server ros-humble-nav2-amcl \
      ros-humble-nav2-controller ros-humble-nav2-planner ros-humble-nav2-behaviors \
      ros-humble-nav2-bt-navigator ros-humble-nav2-lifecycle-manager \
      ros-humble-nav2-costmap-2d ros-humble-nav2-msgs \
      ros-humble-slam-toolbox ros-humble-rtabmap-ros ros-humble-realsense2-camera \
      ros-humble-message-filters ros-humble-cv-bridge ros-humble-image-transport \
      ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-std-msgs \
      ros-humble-std-srvs ros-humble-robot-state-publisher \
      ros-humble-joint-state-publisher \
      ros-humble-naoqi-libqi ros-humble-naoqi-libqicore \
      ros-humble-pepper-meshes ros-humble-nao-meshes \
      libyaml-cpp-dev libopencv-dev libomp-dev libeigen3-dev libboost-all-dev \
      libgl1-mesa-glx libglib2.0-0 libudev-dev libusb-1.0-0 \
      libportaudio2 libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Create the user early: the launcher scripts resolve venvs via $HOME, so the
# venv paths baked in below must match the runtime user's home.
RUN useradd -m -s /bin/bash pepper && mkdir -p ${WS} && chown -R pepper:pepper /home/pepper

# -----------------------------------------------------------------------------
# venv stages - one per node, built in parallel, each from its own lockfile.
# Locks were frozen from the known-good host venvs; the per-package
# requirements.txt files are NOT used (they under-specify torch and list two
# conflicting onnxruntime distributions).
# -----------------------------------------------------------------------------
FROM base AS venv-sound
ARG WS
COPY docker/requirements/sound.lock.txt /tmp/
# torch 2.9.1+cu126 / torchvision 0.24.1+cu126 are not on PyPI.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m venv ${WS}/.venvs/sound && \
    ${WS}/.venvs/sound/bin/pip install --upgrade pip setuptools wheel && \
    ${WS}/.venvs/sound/bin/pip install \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      -r /tmp/sound.lock.txt

FROM base AS venv-conversation
ARG WS
COPY docker/requirements/conversation.lock.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m venv ${WS}/.venvs/conversation && \
    ${WS}/.venvs/conversation/bin/pip install --upgrade pip setuptools wheel && \
    ${WS}/.venvs/conversation/bin/pip install -r /tmp/conversation.lock.txt

FROM base AS venv-tts
ARG WS
COPY docker/requirements/tts.lock.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m venv ${WS}/.venvs/tts_virtual_env && \
    ${WS}/.venvs/tts_virtual_env/bin/pip install --upgrade pip setuptools wheel && \
    ${WS}/.venvs/tts_virtual_env/bin/pip install -r /tmp/tts.lock.txt

# -----------------------------------------------------------------------------
# workspace - clone third-party sources and colcon build.
# Runs in parallel with the venv stages; pepper4dec itself is COPYed from the
# build context (not cloned) so the image matches the checked-out branch.
# -----------------------------------------------------------------------------
FROM base AS workspace
ARG WS
WORKDIR ${WS}

RUN git clone --depth 1 https://github.com/yohatad/naoqi_driver2.git src/naoqi_driver2 && \
    git clone --depth 1 -b ros2 https://github.com/ros-naoqi/libqi.git src/naoqi_libqi && \
    git clone --depth 1 -b ros2 https://github.com/ros-naoqi/libqicore.git src/naoqi_libqicore && \
    git clone --depth 1 https://github.com/ros-naoqi/nao_meshes2.git src/nao_meshes && \
    git clone --depth 1 https://github.com/ros-naoqi/pepper_meshes2.git src/pepper_meshes && \
    git clone --depth 1 https://github.com/BehaviorTree/BehaviorTree.CPP.git src/BehaviorTree.CPP && \
    git clone --depth 1 https://github.com/BehaviorTree/BehaviorTree.ROS2.git src/BehaviorTree.ROS2

# naoqi_bridge_msgs2 is a PRIVATE fork carrying the custom srv definitions
# (SetAudioVolume, StopAudio, UnloadAudioFile) that dec_system depends on;
# ros-naoqi upstream does not have them. Cloned over a forwarded SSH agent so
# no credential is baked into a layer. Requires: docker build --ssh default
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null && \
    git clone --depth 1 git@github.com:yohatad/naoqi_bridge_msgs2.git src/naoqi_bridge_msgs && \
    rm -rf ~/.ssh

COPY . src/pepper4dec

RUN apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y --skip-keys "python3-pip" && \
    rm -rf /var/lib/apt/lists/*

# Parallelism is capped deliberately. naoqi_libqi is boost template-heavy and
# each cc1plus can peak past 1GB; uncapped colcon fans out to nproc (28 here)
# while the venv stages are concurrently resolving multi-GB torch wheels, which
# exhausts RAM and gets the desktop OOM-killed. --parallel-workers caps
# concurrent *packages*; MAKEFLAGS is needed as well to cap make's fan-out
# within each. BUILD_TESTING=OFF skips naoqi_libqi's test suite, whose
# translation units are the largest in the tree and are not needed in the image.
ARG COLCON_JOBS=4
RUN . /opt/ros/humble/setup.sh && \
    I_AGREE_TO_NAO_MESHES_LICENSE=1 I_AGREE_TO_PEPPER_MESHES_LICENSE=1 \
    MAKEFLAGS="-j${COLCON_JOBS}" \
    colcon build --parallel-workers ${COLCON_JOBS} \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

# -----------------------------------------------------------------------------
# runtime - workspace + all three venvs
# -----------------------------------------------------------------------------
FROM workspace AS runtime
ARG WS

COPY --from=venv-sound        ${WS}/.venvs/sound           ${WS}/.venvs/sound
COPY --from=venv-conversation ${WS}/.venvs/conversation    ${WS}/.venvs/conversation
COPY --from=venv-tts          ${WS}/.venvs/tts_virtual_env ${WS}/.venvs/tts_virtual_env
COPY docker/venv_map.sh ${WS}/.venvs/venv_map.sh

# Put ROS2 + the built workspace on each venv's path. The host venvs achieve
# this with editable installs into the colcon build tree; a .pth is the
# container-appropriate equivalent and does not depend on inherited PYTHONPATH.
RUN set -eux; \
    for v in sound conversation tts_virtual_env; do \
      sp="${WS}/.venvs/${v}/lib/python3.10/site-packages"; \
      { echo /opt/ros/humble/lib/python3.10/site-packages; \
        echo /opt/ros/humble/local/lib/python3.10/dist-packages; \
        for d in ${WS}/install/*/local/lib/python3.10/dist-packages; do \
          [ -d "$d" ] && echo "$d"; \
        done; \
      } > "${sp}/ros2_paths.pth"; \
    done

RUN printf '%s\n' \
      '#!/bin/bash' \
      'source /opt/ros/humble/setup.bash' \
      "source ${WS}/install/setup.bash" \
      'export LANG=C.UTF-8 LC_ALL=C.UTF-8' \
      'export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}' \
      '# NOTE: no venv is activated here on purpose -- each node selects its own' \
      '# interpreter through .venvs/venv_map.sh in its launcher script.' \
      > /opt/pepper4dec_setup.sh && chmod +x /opt/pepper4dec_setup.sh

RUN chown -R pepper:pepper /home/pepper

USER pepper
ENV HOME=/home/pepper
WORKDIR ${WS}
ENV BASH_ENV=/opt/pepper4dec_setup.sh
CMD ["/bin/bash"]
