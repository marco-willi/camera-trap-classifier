#!/bin/bash

# install cudnn
tar -zxf cudnn-9.1-linux-x64-v7.1.solitairetheme8
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/
sudo apt-get install libcupti-dev

# Install nVidia docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

# Prepare some directories in home to be attached to docker
cd ~/
mkdir code
mkdir data

# OPTIONAL
# get tensorflow (only if you want to build it yourself)
# git clone https://github.com/tensorflow/tensorflow.git ~/code/tensorflow
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker
# cd code/tensorflow/tensorflow/tools/docker
# modify Dockerfile and add compilation flags for bazel build
# bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2


# launch tensorflow-gpu docker
sudo nvidia-docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-gpu-py3 bash

# launch locally build tensorflow-gpu docker
# sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# install python packages (missing in official tensorflow docker)
pip install dill requests panoptes_client pillow aiohttp keras scikit-image

# save changes
# sudo docker commit docker_id tensorflow/tensorflow:nightly-devel-gpu-py3
