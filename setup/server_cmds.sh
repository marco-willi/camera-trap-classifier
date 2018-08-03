##########################################
# Git commands
##########################################

sudo rm -r ~/code/camera-trap-classifier
git clone https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier


# overwrite local changes in code and pull from master
git fetch --all
git reset --hard origin/master

##########################################
# mount devices (example device is xvdf)
##########################################

# see specific device info
lsblk -o KNAME,TYPE,SIZE,MODEL

# Define file system (only do once)
#sudo file -s /dev/xvdf
#sudo mkfs -t ext4 /dev/xvdf

# Mount device to directory
mkdir ~/data_hdd
sudo mount /dev/xvdf ~/data_hdd


##########################################
# Docker commands
##########################################

# Start Local Nvidia docker
sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# install nightly devel CPU version
sudo docker run -it -v ~/:/host tensorflow/tensorflow:1.9.0-rc1-devel-py3 bash


# Detach from docker with CTRL+P+Q


##########################################
# Misc commands
##########################################

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

# commit changes to docker image
sudo docker commit 244cb559e11b root/tensorflow:latest-devel-gpu-py3

##########################################
# Transferring files
##########################################

# set correct permissions to key file
chmod 400  ~/keys/Machine_Learning.pem

# transfer files from aws to aws
scp -i ~/keys/Machine_Learning.pem /home/ubuntu/data_hdd/west_africa/data/master.tfrecord ubuntu@ec2-34-244-241-168.eu-west-1.compute.amazonaws.com:/home/ubuntu/data_hdd/west_africa/

scp -i ~/keys/Machine_Learning.pem /home/ubuntu/data_hdd/west_africa/experiments/species/data/* ubuntu@ec2-34-244-241-168.eu-west-1.compute.amazonaws.com:/home/ubuntu/data_hdd/west_africa/experiments/species/data/

scp -i ~/keys/Machine_Learning.pem ~/keys/Machine_Learning.pem ubuntu@ec2-34-244-241-168.eu-west-1.compute.amazonaws.com:~/.

scp -i ~/keys/Machine_Learning.pem /home/ubuntu/data_hdd/southern_africa/experiments/species/run_201804032004_incresv2/model_epoch_18.hdf5   ubuntu@ec2-34-248-161-95.eu-west-1.compute.amazonaws.com:/home/ubuntu/data_hdd/southern_africa/experiments/species/run_dummy

# transfer from MSI to AWS
scp -i ~/keys/zv_test_key.pem /home/packerc/will5448/data/tfr_files/all_species_v3/* ubuntu@ec2-52-91-219-250.compute-1.amazonaws.com:/home/ubuntu/data_hdd/ctc/ss/data/species/

/home/packerc/will5448/data/tfr_files/all_species_v3
/home/ubuntu/data_hdd/ctc/ss/data/species
