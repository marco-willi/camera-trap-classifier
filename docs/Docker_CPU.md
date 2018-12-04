# Docker Usage

This document describes how to setup and use Docker. The following commands were executed on an Ubuntu 16.04 base installation on an EC2 instance on AWS.

## Docker Installation

```
sudo apt update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce

# test installation
sudo docker run hello-world
```

## Build Container

This builds the CPU version of the container.

```
# install docker container
mkdir ~/code
cd ~/code
git clone https://github.com/marco-willi/camera-trap-classifier.git
cd camera-trap-classifier
sudo docker build . -f Dockerfile.cpu -t camera_trap_classifier
```

## Start Container

Now we run the container and map /my_data/ from the host computer to /data/ inside the container. It is important to adapt these paths to where the actual data is on the host.

```
# run docker image
# maps /my_data/ on host to /data/ in container
sudo docker run --name ctc -v /my_data/:/data/ -itd camera_trap_classifier
```

## Run Scripts

```
# run scripts with data on host
sudo docker exec ctc ctc.create_dataset_inventory dir -path /data/images \
-export_path /data/dataset_inventory.json

# create directory for tfr-files on host
sudo mkdir /my_data/tfr_files

# create tfr-files
sudo docker exec ctc ctc.create_dataset -inventory /data/dataset_inventory.json \
-output_dir /data/tfr_files/ \
-image_save_side_max 200 \
-split_percent 0.5 0.25 0.25 \
-overwrite


# create directory for log files and model saves
sudo mkdir /my_data/run1 /my_data/save1

# train model
sudo docker exec ctc ctc.train \
-train_tfr_path /data/tfr_files/ \
-val_tfr_path /data/tfr_files/ \
-test_tfr_path /data/tfr_files/ \
-class_mapping_json /data/tfr_files/label_mapping.json \
-run_outputs_dir /data/run1/ \
-model_save_dir /data/save1/ \
-model small_cnn \
-labels class \
-batch_size 16 \
-n_cpus 4 \
-n_gpus 0 \
-buffer_size 16 \
-max_epochs 70 \
-color_augmentation full_randomized

# predict from model
sudo docker exec ctc ctc.predict \
  -image_dir /data/images \
  -results_file /data/output.csv \
  -model_path /data/save1/best_model.hdf5 \
  -class_mapping_json /data/save1/label_mappings.json \
  -pre_processing_json /data/save1/image_processing.json

# export model
sudo mkdir /my_data/save1/my_model_exports/ /my_data/save1/my_estimators/
sudo docker exec ctc ctc.export -model /data/save1/best_model.hdf5 \
  -class_mapping_json /data/save1/label_mappings.json \
  -pre_processing_json /data/save1/image_processing.json \
  -output_dir /data/save1/my_estimators/ \
  -estimator_save_dir /data/save1/my_estimators/keras/

  ```
