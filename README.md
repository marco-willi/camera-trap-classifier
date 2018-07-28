# Camera Trap Image Classifier

**Automatically identify animals in camera trap images by training and applying a deep neural network.**

This repository contains code and documentation to train and apply a convolutional neural network (CNN) for identifying animal species in photographs from camera traps. Please note that this repository will be updated to include more documentation and featuers.

## Example Camera Trap Images and Model Predictions

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/documentation/figures/sample_predictions.png"/>

*This figure shows examples of correctly classified camera trap images.*

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/documentation/figures/sample_predictions_wrong.png"/>

*This figure shows examples of wrongly classified camera trap images (note the lower confidence values).*

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

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/documentation/figures/general_workflow.png"/>

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
python create_dataset_inventory.py csv -path /my_data/dataset_info.csv \
-export_path /my_data/dataset_inventory.json \
-capture_id_field id \
-image_fields image \
-label_fields species count
```

The following code snippet shows how to create a dataset inventory from class directories:
```
python create_dataset_inventory.py dir -path /my_images/all_classes/ \
-export_path /my_data/dataset_inventory.json
```

Note that a json file '/my_data/dataset_inventory.json' is created containing all information.

### 3) Creating the Dataset - TFRecord files

In this step we save all images into large binary '.tfrecord' files which makes it easier to train our models.
The following code snippet shows how that works:

```
python create_dataset.py -inventory /my_data/dataset_inventory.json \
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
python train.py \
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

### 5) Model Use

Finally, we can apply our model on new data. In the following code snippet the program looks for all images
stored in '/my_images/new_images/', including subdirectories.

```
python predict.py -image_dir /my_images/new_images/ \
  -results_file /my_predictions/output.csv \
  -model_path /my_model/save1/pred_model.hdf5 \
  -class_mapping_json /my_model/save1/label_mappings.json \
  -pre_processing_json /my_model/save1/image_processing.json
```

## Installation

The code and the models are based on TensorFlow (https://www.tensorflow.org), a graph-computing software commonly used to implement machine learning models. The installation is relatively easy but can be tricky if an installation with GPU support on a server is required.

We have used python 3.5 (newer versions should work) and Tensorflow 1.9 (older versions don't work).


### Installing from Requirements

The most common way to install all required packages is to create a virtual environment and to use a
requirements.txt file as provided in [setup/](setup/).

```
python3 -m virtualenv ctc
source ctc/bin/activate
pip install -r ./setup/requirements.txt
```

### Windows users with Anaconda

For testing and to use a model for predictions, a local Windows installation can be sufficient. The following
commands allow for a full installation using Anaconda (https://conda.io/docs/user-guide/install/windows.html).
The commands can be executed using, for example, Git BASH (https://gitforwindows.org).

```
# create a new conda environment with name 'ctc'
conda create --no-default-packages -n ctc python=3.5
# activate the environment
source activate ctc
# install tensorflow (according to official documentation)
pip install --upgrade tensorflow
# alternatively: conda install tensorflow
conda install jupyter yaml pyyaml nb_conda pillow h5py
```

### Using a GPU

To train models on large camera trap datasets a GPU is necessary. Besides installing Python and all required modules, nvidia drivers have to be installed on the computer to make use of the GPU (see https://developer.nvidia.com/cuda-downloads and https://developer.nvidia.com/cudnn). An easy option is
to use a disk image that contains all required installations and use that to set up a GPU instance of a cloud provider. Such images are widely available and may be provided by the cloud providers. We created our own image and used AWS to run our models (see below for details).


### Tensorflow GPU Docker installation on AWS
We used Docker (https://www.docker.com/) to run our models on Amazon Web Services (AWS) GPU EC2 instances (https://aws.amazon.com/). The files in /setup/Part_* provide detailled commands on how to install the Tensorflow GPU docker version on a plain Ubuntu base image. It is however not necessary to use Docker - simply installing all modules using the requirements.txt on the GPU server is enough to run all the models. Additional information on how to install
Tensorflow can be found at https://www.tensorflow.org/install/.

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/documentation/figures/server_config.png"/>

*Overview of the AWS setup we used*

## Testing the Code

Following commands should run without error:

```
python -m unittest discover test/data
python -m unittest discover test/training
```

There are some manual tests in 'test/manual_tests' that require to create directories to and provide some simple data (image directories).

## Exporting a Model

**WARNING**: The following is only possible with Tensorflow 1.9.

To export a model for later deployment we can use the following code:

```
python export.py -model /my_experiment/model_save_dir/prediction_model.hdf5 \
-class_mapping_json /my_experiment/model_save_dir/label_mappings.json \
-pre_processing_json /my_experiment/model_save_dir/pre_processing.json \
-output_dir /my_experiment/my_model_exports/ \
-estimator_save_dir /my_experiment/my_estimators/
```

## Deploying a Model

To deploy a model on a server we refer to this documentation:
[How to deploy a model](deploy/README.md)


## Acknowledgements

This code is based on work conducted in the following study:

*Identifying Animal Species in Camera Trap Images using Deep Learning and Citizen Science, 2018, submitted*

Authors: Marco Willi, Ross Tyzack Pitman, Anabelle W. Cardoso, Christina Locke, Alexandra Swanson, Amy Boyer, Marten Veldthuis, Lucy Fortson


The ResNet models are based on the implementation provided here:
https://github.com/raghakot/keras-resnet
