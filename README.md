<img src="https://s3-eu-west-1.amazonaws.com/pantherabucketleopard1/27_2016/2016-11-14_09-05-21-CAM51271.jpg"  width="150" height="125" style="float: right" />
# Camera Trap Image Classifier
Automatically identify animals in camera trap images by training and applying a deep neural network.

## Overview

This repository contains code and documentation to train and apply a convolutional neural network (CNN) for identifying animal species in photographs from camera traps.

## Pre-Requisites

To use this code following pre-requisites must be met:

1. Camera trap images (jpeg / jpg) with labels
2. Access to computer/server with graphics processing units (GPUs) for model training (e.g. AWS account)
3. Some (little) knowledge of Unix

## General Process

The following steps are required to train a model:

1. Organizing image and label data according to different options (see below).
2. Create a dataset inventory which is a file that contains all links and labels of images
3. Create training/test/validation data from a dataset inventory. All images are saved into large 'tfrecord' files.
4. Train a model.
5. Apply a model on new data.

### Data Preparation

The first thing is to organize the image and label data. There are several options:

1. Save images into class-specific image directories (image names are arbitrary).
```
root_dir:
  - elephant
    - elephant1.jpg
    - elephant2.jpg
    - ...
  - zebra
      - zebra1.jpg
      - zebra2.jpg
      - ...
```
2. Create a csv file that contains all labels and links to images.
```
id,image,species,count
1,/my_images/image1.jpg,elephant,2
2,/my_images/image2.jpg,elephant,10
3,/my_images/image3.jpg,lion,1
4,/my_images/image4.jpg,zebra,10
```
The advantage of using a csv file is that more than one label can be provided. In this example species and count.


### Creating a Dataset Inventory

In this step a common data structure is created regardless of how the data preparation was done. The following code snippet shows how to create a dataset inventory based on a csv file.

```
python create_dataset_inventory.py csv -path /my_data/dataset_info.csv \
-export_path /my_data/dataset_inventory.json \
-capture_id_field id \
-image_fields image \
-label_fields species count
```

The following snippet shows how to create a dataset inventory from class directories:
```
python create_dataset_inventory.py dir -path /my_images/all_classes/ \
-export_path /my_data/dataset_inventory.json
```

Note that a json file '/my_data/dataset_inventory.json' is created containing all information.

### Creating the Dataset - TFRecord files

In this step we save all images into large binary '.tfrecord' files which makes it easier to train our models.
The following code snippet shows how that works:

```
python create_dataset.py -inventory /my_data/dataset_inventory.json \
-output_dir /my_data/tfr_files/ \
-image_save_side_max 200 \
-split_percent 0.9 0.05 0.05 \
-overwrite
```

### Model Training

In the next step we train our model. The following code snippet shows an example:

```
python train_model.py \
-train_tfr_path /my_data/tfr_files/ \
-val_tfr_path /my_data/tfr_files/ \
-test_tfr_path /my_data/tfr_files/ \
-class_mapping_json /my_data/tfr_files/label_mapping.json \
-run_outputs_dir /my_model/run1/ \
-model_save_dir /my_model/save1/ \
-model ResNet18 \
-labels species count \
-batch_size 128 \
-n_cpus 4 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 70
```

### Model Use

Finally, we can apply our model on new data. In the following code snippet the program looks for all images
stored in /my_images/new_images/, including subdirectories.

```
python main_prediction.py -image_dir /my_images/new_images/ \
  -results_file /my_predictions/output.csv \
  -model_path /my_model/save1/pred_model.hdf5 \
  -class_mapping_json /my_model/save1/label_mappings.json \
  -pre_processing_json /my_model/save1/image_processing.json
```

## Installation

The code and the models are based on TensorFlow (https://www.tensorflow.org) a graph-computing software, commonly
used to implement machine learning models. The installation is reltively easy but can be tricky if an installation with
GPU support on a serve is required.

We have used python 3.5 and Tensorflow 1.6 (newer/older version may work as well).

### Tensorflow GPU Docker installation on AWS
The files in /setup/Part_* provide detailled commands on how to install the Tensorflow GPU docker version.
Alternatively, https://www.tensorflow.org/install/ provides guidelines on how to install Tensorflow (GPU) version.

### Windows users with Anaconda

For testing and to use a model for predictions a local Windows installation can be sufficient. The following
commands allow for a full installation using Anaconda:

```
# create a new conda environment
conda create --no-default-packages -n ctc python=3.5
source activate ctc
pip install --upgrade tensorflow
conda install jupyter yaml nb_conda pillow h5py
```

## Testing the Code

Following commands should run without error:

```
python -m unittest discover test/data_processing
python -m unittest discover test/training
```

## Acknowledgements

This code is based on work conducted in following study:

*Identifying Animal Species in Camera Trap Images using Deep Learning and Citizen Science, 2018, in preparation*

Authors: Marco Willi, Ross Tyzack Pitman, Anabelle W. Cardoso, Christina Locke, Alexandra Swanson, Amy Boyer, Marten Veldthuis, Lucy Fortson


The ResNet models are based on the implementation provided here:
https://github.com/raghakot/keras-resnet
