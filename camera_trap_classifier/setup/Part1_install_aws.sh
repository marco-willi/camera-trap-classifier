#!/bin/bash

# update
sudo apt-get update

# install some basics
sudo apt-get install -y build-essential git python-pip libfreetype6-dev
 libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib
 libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot
 linux-headers-generic linux-image-extra-virtual unzip python-numpy
 swig python-pandas python-sklearn unzip wget pkg-config zip g++ zlib1g-dev
 screen emacs24

# install docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce

# install CUDA
sudo apt-get install linux-headers-$(uname -r)
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
# (may need to install key according to terminal output and repeat)

sudo apt-get update
sudo apt-get install cuda
export PATH=$PATH:/usr/local/cuda-9.1/bin
export LD_LIBRARY_PATH=:/usr/local/cuda-9.1/lib64

# Reboot machine
sudo reboot

# Verify installation
# http://docs.nvidia.com/cuda/cuda-installation-guide-linux
cat /proc/driver/nvidia/version
