sudo rm -r ~/code/camera-trap-classifier
git clone https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier


# overwrite local changes in code and pull from master
git fetch --all
git reset --hard origin/master

# mount devices (example device is xvdf)
# see specific device info
#sudo file -s /dev/xvdf
#sudo mkfs -t ext4 /dev/xvdf
mkdir ~/data_hdd
sudo mount /dev/xvdf ~/data_hdd


# Local Nvidia docker
sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# Detach from docker with CTRL+P+Q

# Monitor GPU utilization
# nvidia-smi -l 1

# run tests
python3 -m unittest discover test/

# fix multi gpu driver on host
sudo nvidia-modprobe -u -c=0
