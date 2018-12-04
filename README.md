[![DOI](https://zenodo.org/badge/125009121.svg)](https://zenodo.org/badge/latestdoi/125009121)
# Camera Trap Image Classifier

**Automatically identify animals in camera trap images by training and applying a deep neural network.**

This repository contains code and documentation to train and apply a convolutional neural network (CNN) for identifying animal species in photographs from camera traps. Please note that this repository will be updated to include more documentation and featuers.

## Example Camera Trap Images and Model Predictions

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/sample_predictions.png"/>

*This figure shows examples of correctly classified camera trap images.*

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/sample_predictions_wrong.png"/>

*This figure shows examples of wrongly classified camera trap images (note the lower confidence values).*

## Pre-Requisites

To use this code following pre-requisites must be met:

1. Camera trap images (jpeg / jpg / png /bmp) with labels
2. Server with graphics processing units (GPUs) for model training (e.g. AWS account, supercomputing institute)
3. Some Unix knowledge

## Installation

The code and the models are based on TensorFlow (https://www.tensorflow.org), a graph-computing software commonly used to implement machine learning models. The installation is relatively easy but can be tricky if an installation with GPU support on a server is required.

### Installation from GitHub

The software and all dependencies can be installed with this command:
```
pip install git+git://github.com/marco-willi/camera-trap-classifier.git#egg=camera_trap_classifier[tf]
```

To install the GPU version use this command:
```
pip install git+git://github.com/marco-willi/camera-trap-classifier.git#egg=camera_trap_classifier[tf-gpu]
```

The software can then be used from the command line (see below for more details):
```
ctc.create_dataset_inventory --help
ctc.create_dataset --help
ctc.train --help
ctc.predict --help
ctc.export --help
```


### Anaconda Users

The following commands allow for a full installation using Anaconda (https://conda.io/docs/user-guide/install/index.html).
The commands can be executed, for example on Windows, using Git BASH (https://gitforwindows.org) or using the terminal on Unix systems.

```
# create a new conda environment with name 'ctc'
conda create --no-default-packages -n ctc python=3.5
# activate the environment
source activate ctc
# install tensorflow (non-GPU or GPU version)
conda install tensorflow=1.12.0
# conda install tensorflow-gpu=1.12.0
conda install pyyaml pillow
```

### Docker

The software can also be installed using Docker. There are two versions, a CPU and a GPU Tensorflow installation.

https://docs.docker.com/get-started/

To build the container:
```
docker build . -f Dockerfile.cpu -t camera_trap_classifier
```

To run commands inside the container:
```
docker run --name ctc -v /my_data/:/data/ -itd camera_trap_classifier
docker exec ctc ctc.train --help
docker exec ctc ctc.train --predict
```

A detailed example on how to install Docker and run scripts can be found here:
[Install and use CPU Docker](docs/Docker_CPU.md)

### Using a GPU

To train models on large camera trap datasets a GPU is necessary. Besides installing Python and all required modules, nvidia drivers have to be installed on the computer to make use of the GPU. More details can be found here: https://www.tensorflow.org/install/gpu

Alternatively, cloud providers often provide pre-configured servers with all installations.


## General Process

The following steps are required to train a model:

1. Organizing image and label data according to different options (see below).
2. Create a dataset inventory which is a file that contains all links and labels of images
3. Create training/test/validation data from a dataset inventory. All images are saved into large 'tfrecord' files.
4. Train a model.
5. Apply a model on new data.

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/general_workflow.png"/>

*Overview of the process*


### 1) Data Preparation

The first thing is to organize the image and label data. There are several options:

**Option 1**: Save images into class-specific image directories (image names are arbitrary).
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
**Option 2**: Create a csv file that contains all labels and links to images.

The advantage of using a csv file is that more than one label can be provided. In this example species and count.

```
id,image,species,count
1,/my_images/image1.jpg,elephant,2
2,/my_images/image2.jpg,elephant,10
3,/my_images/image3.jpg,lion,1
4,/my_images/image4.jpg,zebra,10
```

Multiple images can be grouped into one capture event. During model training a random image will be chosen, also during the evaluation. Other, more sophisticated ways to handle multi-image capture events can be implemented.

```
id,image1,image2,species,count
1,/my_images/image1a.jpg,/my_images/image1b.jpg,elephant,2
2,/my_images/image2a.jpg,/my_images/image2b.jpg,elephant,10
3,/my_images/image3a.jpg,,lion,1
4,/my_images/image4a.jpg,/my_images/image4b.jpgzebra,10
```

Multiple observations per capture event can be grouped. Note that modelling multi-label multi-class classification is not supported. However, the data will be processed and stored to TFRecord files but only one observation is chosen during model training and evaluation.

```
id,image,species,count
1,/my_images/image1.jpg,elephant,2
1,/my_images/image1.jpg,lion,3
2,/my_images/image2.jpg,elephant,10
3,/my_images/image3.jpg,lion,1
4,/my_images/image4.jpg,zebra,10
4,/my_images/image4.jpg,wildebeest,2
```


### 2) Creating a Dataset Inventory

In this step a common data structure is created regardless of how the data preparation was done. The following code snippet shows how to create a dataset inventory based on a csv file.

```
ctc.create_dataset_inventory csv -path /my_data/dataset_info.csv \
-export_path /my_data/dataset_inventory.json \
-capture_id_field id \
-image_fields image \
-label_fields species count
```

The following code snippet shows how to create a dataset inventory from class directories:
```
ctc.create_dataset_inventory dir -path /my_images/all_classes/ \
-export_path /my_data/dataset_inventory.json
```

Note that a json file '/my_data/dataset_inventory.json' is created containing all information.

### 3) Creating the Dataset - TFRecord files

In this step we save all images into large binary '.tfrecord' files which makes it easier to train our models.
The following code snippet shows how that works:

```
ctc.create_dataset -inventory /my_data/dataset_inventory.json \
-output_dir /my_data/tfr_files/ \
-image_save_side_max 200 \
-split_percent 0.9 0.05 0.05 \
-overwrite
```

See the function documentations for options regarding how to parallelize / speed up
the processing for large datasets.

### 4) Model Training

In the next step we train our model. The following code snippet shows an example:

```
ctc.train \
-train_tfr_path /my_data/tfr_files/ \
-val_tfr_path /my_data/tfr_files/ \
-test_tfr_path /my_data/tfr_files/ \
-class_mapping_json /my_data/tfr_files/label_mapping.json \
-run_outputs_dir /my_model/run1/ \
-model_save_dir /my_model/save1/ \
-model ResNet18 \
-labels species count \
-labels_loss_weights 1 0.5 \
-batch_size 128 \
-n_cpus 4 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 70 \
-color_augmentation full_randomized
```

Use the following command for more help about all the options:
```
ctc.train --help
```

### 5) Model Use

Finally, we can apply our model on new data. In the following code snippet the program looks for all images
stored in '/my_images/new_images/', including subdirectories.

```
ctc.predict -image_dir /my_images/new_images/ \
  -results_file /my_predictions/output.csv \
  -model_path /my_model/save1/best_model.hdf5 \
  -class_mapping_json /my_model/save1/label_mappings.json \
  -pre_processing_json /my_model/save1/image_processing.json
```


## Testing the Code

Following commands should run without error and test a part of the code:

```
cd camera_trap_classifier
python -m unittest discover test/data
python -m unittest discover test/training
```

The following script tests all components of the code end-to-end using images from a directory:

```
# 1) adapt the parameters in ./test/full_tests/from_image_dir_test.sh
# 2) create all the directories as referenced in the script
# 3) run the script
cd camera_trap_classifier
./test/full_tests/from_image_dir_test.sh
```

The following script tests training from data with multiple images per capture event:

```
# 1) adapt the parameters in ./test/full_tests/complete_cats_vs_dogs_test_multi.sh
# 2) create all the directories as referenced in the script
# 3) run the script
cd camera_trap_classifier
./test/full_tests/complete_cats_vs_dogs_test_multi.sh
```

There are some manual tests in 'test/manual_tests' for different components that can be run interactively.

## Exporting a Model

**WARNING**: The following is only possible with Tensorflow 1.9.

To export a model for later deployment we can use the following code:

```
ctc.export -model /my_experiment/model_save_dir/prediction_model.hdf5 \
-class_mapping_json /my_experiment/model_save_dir/label_mappings.json \
-pre_processing_json /my_experiment/model_save_dir/pre_processing.json \
-output_dir /my_experiment/my_model_exports/ \
-estimator_save_dir /my_experiment/my_estimators/
```

## Deploying a Model

To deploy a model on a server we refer to this documentation:
[How to deploy a model](docs/deploy/README.md)


## Acknowledgements

This code is based on work conducted in the following study:

*Identifying Animal Species in Camera Trap Images using Deep Learning and Citizen Science, 2018, Methods in Ecology and Evolution*

Authors: Marco Willi, Ross Tyzack Pitman, Anabelle W. Cardoso, Christina Locke, Alexandra Swanson, Amy Boyer, Marten Veldthuis, Lucy Fortson


The ResNet models are based on the implementation provided here:
https://github.com/raghakot/keras-resnet
