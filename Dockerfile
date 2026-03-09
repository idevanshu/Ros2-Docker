FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    nano git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install onnxruntime-gpu numpy

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
WORKDIR /ros2_ws
