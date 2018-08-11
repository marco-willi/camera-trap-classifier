# Panthera / Snapshot Safari Use Case

This document describes how data for Panthera and Snapshot Safari were modeled on AWS EC2 instances.


## AWS EC2 Instance Preparation

1. Spin up an EC2 instance (p3.2xlarge with 1 GPU or p3.8xlarge with 4 GPUs)
2. Load snapshot with tfrecord data to an SSD (same avail zone as instance)
3. Attach SSD to EC2 instance
4. Connect to EC2 instance
5. Get code to instance:

```
git clone https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier
```

6. (Optional) - run following command for p3.x instances
```
sudo nvidia-modprobe -u -c=0
```

7. Start Tensorflow Docker container
```
sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash
```

8. Go to code directory from within Docker
```
cd /host/code/camera-trap-classifier
```

9. Run the codes below.

10. Detach from Docker with CTRL + P Q

11. Various server commands:
```
# list drives
lsblk -o KNAME,TYPE,SIZE,MODEL

# attach ssd
sudo mount /dev/xvdf ~/data_hdd

# run docker
sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# Monitor GPU utilization
nvidia-smi -l 1

# fix multi gpu driver on host
sudo nvidia-modprobe -u -c=0
```

## Data Organization on the EC2 instance

The data was organized in the following way:

### TFRecord files

For example, the tfrecord fiels for southern africa:
```
~/data_hdd/data/tfr_files/species/southern_africa
```

### Run data

The following folders are to save model data and logfiles to.
```
~/data_hdd/runs/species/sa_and_ss/outputs_all_labels/
~/data_hdd/runs/species/sa_and_ss/saves_all_labels/
```

### Dataset inventories

The following folder is for saving dataset inventories.
```
~/data_hdd/data/inventories
```

## Running the Codes

### 1) Data Preparation

Prepare the data using a csv containing columns for:
- image paths
- labels
- split_name
- meta_data

Split name will be used to separate different records into different tfrecord files.


### 2) Create Data Inventory

The following command creates a dataset inventory.

```
python create_dataset_inventory.py csv -path /host/data_hdd/southern_africa/data/inventory_list_20180806.csv \
-export_path /host/data_hdd/tfr_data/inventories/dataset_inventory_southern_africa.json \
-capture_id_field capture_id \
-image_fields dir \
-label_fields empty species count standing resting moving eating interacting babies species_panthera \
-meta_data_fields survey image location split_name
```

### 3) Create TFRecord Files / Dataset

The following command prepares data containing only species images. Set the number of 'processes_images_in_parallel_n_processes' to the number of available cpu cores.

```
python create_dataset.py -inventory /host/data_hdd/tfr_data/inventories/dataset_inventory_southern_africa.json \
-output_dir /host/data_hdd/tfr_data/species/southern_africa/ \
-image_save_side_max 500 \
-split_by_meta split_name \
-remove_label_name empty empty \
-remove_label_value empty vehicle \
-image_root_path /host/data_hdd/southern_africa/data/ \
-process_images_in_parallel \
-process_images_in_parallel_size 640 \
-processes_images_in_parallel_n_processes 16 \
-max_records_per_file 5000
```

The following command prepares data for the empty model.
```
python create_dataset.py -inventory /host/data_hdd/tfr_data/inventories/dataset_inventory_southern_africa.json \
-output_dir /host/data_hdd/tfr_data/empty_or_not/southern_africa/ \
-image_save_side_max 500 \
-split_by_meta split_name \
-balanced_sampling_min \
-balanced_sampling_label empty \
-image_root_path /host/data_hdd/southern_africa/data/ \
-process_images_in_parallel \
-process_images_in_parallel_size 640 \
-processes_images_in_parallel_n_processes 16 \
-max_records_per_file 5000
```


### 4) Train a Model

The species model (trained on p3.8xlarge)

```
python3 train.py \
-train_tfr_path /host/data_hdd/data/tfr_files/species/ \
-train_tfr_pattern train \
-val_tfr_path /host/data_hdd/data/tfr_files/species/ \
-val_tfr_pattern val \
-test_tfr_path /host/data_hdd/data/tfr_files/species/ \
-test_tfr_pattern test \
-class_mapping_json /host/data_hdd/data/common_label_mappings/species/label_mapping.json \
-run_outputs_dir /host/data_hdd/runs/species/sa_and_ss/outputs_all_labels/ \
-model_save_dir /host/data_hdd/runs/species/sa_and_ss/saves_all_labels/ \
-model InceptionResNetV2 \
-labels species count standing resting moving eating interacting babies \
-batch_size 128 \
-n_cpus 32 \
-n_gpus 4 \
-buffer_size 32768 \
-max_epochs 50 \
-color_augmentation full_randomized \
-ignore_aspect_ratio > out.log &
```

Use the following command to monitor the training progress:
```
tail -f out.log
```

The empty model (Trained on p3.2xlarge)

```
python3 train.py \
-train_tfr_path /host/data_hdd/data/tfr_files/empty_or_not/ \
-train_tfr_pattern train \
-val_tfr_path /host/data_hdd/data/tfr_files/empty_or_not/ \
-val_tfr_pattern val \
-test_tfr_path /host/data_hdd/data/tfr_files/empty_or_not/ \
-test_tfr_pattern test \
-class_mapping_json /host/data_hdd/data/common_label_mappings/empty_or_not/label_mapping.json \
-run_outputs_dir /host/data_hdd/runs/empty_or_not/sa_and_ss/outputs/ \
-model_save_dir /host/data_hdd/runs/empty_or_not/sa_and_ss/saves/ \
-model ResNet18 \
-labels empty \
-batch_size 256 \
-n_cpus 8 \
-n_gpus 1 \
-buffer_size 32768 \
-max_epochs 30 \
-color_augmentation full_randomized \
-ignore_aspect_ratio > out.log &
```


### 5) Transfer-Learning on New Data

This is an example of how a new model could be trained with potentially new species classes.

1. Prepare a new csv
2. Create Dataset Inventory
3. Create TFRecord Files
4. Train model like this (e.g. for west_africa):

```
python3 train.py \
-train_tfr_path /host/data_hdd/data/tfr_files/species/west_africa/ \
-train_tfr_pattern train \
-val_tfr_path /host/data_hdd/data/tfr_files/species/west_africa/ \
-val_tfr_pattern val \
-test_tfr_path /host/data_hdd/data/tfr_files/species/west_africa/ \
-test_tfr_pattern test \
-class_mapping_json /host/data_hdd/data/tfr_files/species/west_africa/label_mapping.json \
-run_outputs_dir /host/data_hdd/runs/species/west_africa/outputs/ \
-model_save_dir /host/data_hdd/runs/species/west_africa/saves/ \
-model InceptionResNetV2 \
-labels species_panthera count \
-batch_size 128 \
-n_cpus 32 \
-n_gpus 4 \
-buffer_size 32768 \
-max_epochs 50 \
-color_augmentation full_randomized \
-transfer_learning \
-model_to_load /host/data_hdd/runs/species/sa_and_ss/saves_all_labels/best_model.hdf5 \
-ignore_aspect_ratio > out.log &
```


### 6) Fine-Tuning using More / New Data.

This is an example of how a trained model could be adapted to new data using the same output classes.

1. Prepare a new csv
2. Create Dataset Inventory
3. Create TFRecord Files
4. Train model like this (e.g. for west_africa):

```
python3 train.py \
-train_tfr_path /host/data_hdd/data/tfr_files/species/ \
-train_tfr_pattern train \
-val_tfr_path /host/data_hdd/data/tfr_files/species/ \
-val_tfr_pattern val \
-test_tfr_path /host/data_hdd/data/tfr_files/species/ \
-test_tfr_pattern test \
-class_mapping_json /host/data_hdd/data/common_label_mappings/species/label_mapping.json \
-run_outputs_dir /host/data_hdd/runs/species/west_africa/outputs/ \
-model_save_dir /host/data_hdd/runs/species/west_africa/saves/ \
-model InceptionResNetV2 \
-labels species count standing resting moving eating interacting babies \
-batch_size 128 \
-n_cpus 32 \
-n_gpus 4 \
-buffer_size 32768 \
-max_epochs 50 \
-color_augmentation full_randomized \
-continue_training \
-model_to_load /host/data_hdd/runs/species/sa_and_ss/saves_all_labels/best_model.hdf5 \
-ignore_aspect_ratio > out.log &
```

Use the following additional option if the model crashes (this can happen if different settings are chosen).
Alternatively, use '-transfer_learning' instead of '-continue_training'.
```
-rebuild_model
-learning_rate 0.001
```

Additionally, to avoid overfitting if the new dataset is not that large: mix in tfrecord files
from other locations.



### 7) Continue Training After Server Crash

This is an example of continuing to train the species model after the server crashed after completing
10 epochs (the lastest model in 'outputs_all_labels' was model_epoch_10_loss_4.41.hdf5). Note that we start the
training with 'starting_epoch' equals 10 (0-based index, thus actually the 11th epoch). The path in 'model_to_load'
specifies the location to search for the most recent model to load and to continue training from. Note also that
we had to specify '-rebuild_model' because it would crash with an out of memory error (using a 4 GPU instance).


```
python3 train.py \
-train_tfr_path /host/data_hdd/data/tfr_files/species/ \
-train_tfr_pattern train \
-val_tfr_path /host/data_hdd/data/tfr_files/species/ \
-val_tfr_pattern val \
-test_tfr_path /host/data_hdd/data/tfr_files/species/ \
-test_tfr_pattern test \
-class_mapping_json /host/data_hdd/data/common_label_mappings/species/label_mapping.json \
-run_outputs_dir /host/data_hdd/runs/species/sa_and_ss/outputs_all_labels/ \
-model_save_dir /host/data_hdd/runs/species/sa_and_ss/saves_all_labels/ \
-model InceptionResNetV2 \
-labels species count standing resting moving eating interacting babies \
-batch_size 128 \
-n_cpus 32 \
-n_gpus 4 \
-buffer_size 32768 \
-max_epochs 50 \
-starting_epoch 10 \
-color_augmentation full_randomized \
-continue_training \
-rebuild_model \
-model_to_load /host/data_hdd/runs/species/sa_and_ss/outputs_all_labels/ \
-ignore_aspect_ratio
```
