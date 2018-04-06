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

# install python virtual environment
conda create -n ct3 python=3.5
conda list -e > requirements.txt
conda install --yes --file requirements.txt



# transfer files from aws to aws
chmod 400  ~/keys/Machine_Learning.pem
scp -i ~/keys/Machine_Learning.pem /home/ubuntu/data_hdd/west_africa/data/master.tfrecord ubuntu@ec2-34-244-241-168.eu-west-1.compute.amazonaws.com:/home/ubuntu/data_hdd/west_africa/

scp -i ~/keys/Machine_Learning.pem /home/ubuntu/data_hdd/west_africa/experiments/species/data/* ubuntu@ec2-34-244-241-168.eu-west-1.compute.amazonaws.com:/home/ubuntu/data_hdd/west_africa/experiments/species/data/


scp -i ~/keys/Machine_Learning.pem ~/keys/Machine_Learning.pem ubuntu@ec2-34-244-241-168.eu-west-1.compute.amazonaws.com:~/.


scp -i ~/keys/Machine_Learning.pem /home/ubuntu/data_hdd/southern_africa/experiments/species/run_201804032004_incresv2/model_epoch_18.hdf5   ubuntu@ec2-34-248-161-95.eu-west-1.compute.amazonaws.com:/home/ubuntu/data_hdd/southern_africa/experiments/species/run_dummy
