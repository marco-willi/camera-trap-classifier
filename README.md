[![DOI](https://zenodo.org/badge/125009121.svg)](https://zenodo.org/badge/latestdoi/125009121)
# Camera Trap Image Classifier

**Automatically identify animals in camera trap images by training and applying a deep neural network.**

This repository contains code and documentation to train and apply convolutional neural networks (CNN) for identifying animal species in photographs from camera traps. Please note that this repository will be updated to include more documentation and features.

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/sample_predictions.png"/>

*This figure shows examples of correctly classified camera trap images.*

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/sample_predictions_wrong.png"/>

*This figure shows examples of wrongly classified camera trap images (note the lower confidence values).*

## Features
1. Step-by-step instructions on how to train powerful models in the cloud ([AWS example](docs/Docker_GPU.md))
2. Modelling of capture events with multiple images (store, process, and predict together)
3. Multi-output modelling: model species, behaviors, and any other label at the same time.
4. A large variety of options: models, data augmentation, installation, data-prep, transfer-learning, and more.
5. Tested approach: This code is currently in use and is being developed further.

## Pre-Requisites

To use this code following pre-requisites must be met:

1. Camera trap images (jpeg / jpg / png /bmp) with labels
2. Computer with GPUs for model training (e.g. AWS account, supercomputing institute)
3. Some Unix knowledge

## Installation

The code has been implemented in Python (https://www.python.org) and is based on TensorFlow (https://www.tensorflow.org), a graph-computing software commonly used to implement machine learning models. The installation is relatively easy but can be tricky if an installation with GPU support on a server is required. We recommend using Docker on a GPU instance of a cloud provider ([AWS example](docs/Docker_GPU.md)).

### Installation from GitHub

The software and all dependencies can be installed with this command (CPU version):
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
pip install git+git://github.com/marco-willi/camera-trap-classifier.git
```


### Docker

The software can also be installed using Docker (https://docs.docker.com/get-started/). There are two versions, a CPU and a GPU Tensorflow installation.

To build the (GPU or CPU) container:
```
docker build . -f Dockerfile.gpu -t camera-trap-classifier:latest-gpu
docker build . -f Dockerfile.cpu -t camera-trap-classifier:latest-cpu
```

To start the container:
```
docker run --name ctc -v /my_data/:/data/ -itd camera-trap-classifier:latest-gpu
```

To run commands inside the container:
```
docker exec ctc ctc.train --help
docker exec ctc ctc.train --predict
```

#### Example for using Docker on AWS

We have run our models on AWS EC2 instances using Docker. A detailed example on how to install Docker and run scripts can be found here:

[Install and use CPU Docker](docs/Docker_CPU.md)

[Install and use GPU Docker](docs/Docker_GPU.md)


#### Singularity Containers

Some environments may not allow the use of Docker (e.g. super computing institutes). Sometimes, Singularity containers are available. Here is an example:

[Singularity Example](docs/Singularity.md)

### Using a GPU

To train models on large camera trap datasets a GPU is necessary. Besides installing Python and all required modules, nvidia drivers have to be installed on the computer to make use of the GPU. More details can be found here: https://www.tensorflow.org/install/gpu

Alternatively, cloud providers often provide pre-configured servers with all installations. [The Docker installation with GPU on AWS](docs/Docker_GPU.md) provides straight forward instructions (see the example with Cats vs Dogs).

## General Process

The following steps are required to train a model:

1. Organizing image and label data according to different options.
2. Create a dataset inventory that contains all links and labels of the images.
3. Create training/test/validation data from a dataset inventory.
4. Train a model.
5. Apply a model on new data.


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

The following code snippet shows how to create a dataset inventory from class directories (in that case the label will be refered to as 'class' -- see model training section):
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
-image_save_side_smallest 400 \
-split_percent 0.9 0.05 0.05 \
-overwrite \
-process_images_in_parallel \
-process_images_in_parallel_size 320 \
-processes_images_in_parallel_n_processes 4 \
-image_save_quality 75 \
-max_records_per_file 5000
```

See the function documentations for options regarding details on how to parallelize / speed up
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
-max_epochs 70
```

Use the following command for more help about all the options:
```
ctc.train --help
```

If the data inventory was created from class directories (Option 1) the default and only label will be 'class'. This means the model has to be trained with the following 'labels' parameter:
```
-labels class \
```
See also the example in: [AWS Example](docs/Docker_CPU.md)

See section 'Data Augmentation' for more details about how to modify model training.

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

We can also use a csv for our predictions to process and aggregate multiple images for a capture event. For example, consider the following csv:

```
id,image1,image2,image3
1,/unknown1_a.jpeg,/unknown1_b.jpeg,/unknown1_c.jpeg
2,/unknown2_a.jpeg,/unknown2_b.jpeg,/unknown2_c.jpeg
```

With the following command the images of a capture event are each passed through the model and at the end aggregated on id (csv_id_col) level using the specified aggregation mode. In this case the predictions of all classes over all images of
a capture event are averaged. The top-prediction is then determined on that aggregated metric.

```
ctc.predict \
  -csv_path /my_images/new_images/inventory.csv \
  -csv_id_col id \
  -csv_images_cols image1 image2 image3 \
  -export_file_type csv \
  -results_file /my_preds/preds.csv \
  -model_path /my_model/save1/best_model.hdf5 \
  -class_mapping_json /my_model/save1/label_mappings.json \
  -pre_processing_json /my_model/save1/image_processing.json \
  -aggregation_mode mean
```

The predictions may look like that then:
```
"id","label","prediction_top","confidence_top","predictions_all"
"1","species","cat","0.5350","{'cat': '0.5350', 'dog': '0.4650'}"
"1","standing","1","0.5399","{'0': '0.4601', '1': '0.5399'}"
"2","species","cat","0.5160","{'cat': '0.5160', 'dog': '0.4840'}"
"2","standing","1","0.5064","{'0': '0.4936', '1': '0.5064'}"
```

Note: The json export (export_file_type json) contains also the unaggregated predictions of each image.

### Data Augmentation

To avoid overfitting, images are randomly transformed in various ways during model training. The following options are available (defaults):

```
-color_augmentation (full_randomized)
-crop_factor (0.1)
-zoom_factor (0.1)
-rotate_by_angle (5)
-randomly_flip_horizontally (True)
-preserve_aspect_ratio (False)
```
See ctc.train --help for more details.

The default values produce following examples:

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/data_augmentation_train_default.png"/>

*This figure shows examples of randomly gerenated images using default data augmentation parameters*

Important to note is that heavy data augmentation is quite expensive. Training a small model on 2 GPUs with a batch size of 256 required roughly 20 CPUs to keep the GPUs busy. This is less of a problem for larger models since the GPUs will be the
bottleneck. Check CPU usage with the 'top' command. We also recommend the nvidia tool 'nvidia-smi -l 1' to check the GPU usage during model training (it should be near 100% all the time). If performance is a problem, set rotate_by_angle to 0, followed by zooming.

### Experimental Feature - Grayscale Stacking

To make better use of motion information contained in capture events with multiple images there is an option to stack up to three images into a single RGB image. The option can be activated using:

```
-image_choice_for_sets grayscale_stacking
```

This will apply the following transformations to each image during model training:
1. Convert each image of a capture event (set of image) to grayscale
2. (Blurr each image with a Gaussian filter - DEPRECATED due to poor performance on CPU)
3. Stack all images of the set in temporal order to an RGB image (e.g. first image goes into the 'red' channel). If there are fewer than 3 images in some sets, the last image is repeated.

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/master/docs/figures/data_augmentation_grayscale_stacking.png"/>

*This figure shows examples of grayscale stacking on sets with 1 or 3 images*

This feature is experimental in the sense that it has not yet been thoroughly tested. We think that this option may benefit models which have to identify the presence of animals -- especially if there are small animals that are difficult to see without the motion information.


## Testing the Code

Following commands should run without error and test a part of the code:

```
cd camera_trap_classifier
python -m unittest discover test/data
python -m unittest discover test/predicting
# the next one runs long
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

## Exporting a Model (NEEDS UPDATE - DOES NOT WORK CORRECTLY)

To export a model for later deployment we can use the following code:

```
ctc.export -model /my_experiment/model_save_dir/prediction_model.hdf5 \
-class_mapping_json /my_experiment/model_save_dir/label_mappings.json \
-pre_processing_json /my_experiment/model_save_dir/pre_processing.json \
-output_dir /my_experiment/my_model_exports/ \
-estimator_save_dir /my_experiment/my_estimators/
```

## Deploying a Model (NEEDS UPDATE - DOES NOT WORK CORRECTLY)

Experimental - Needs update.

To deploy a model on a server we refer to this documentation:
[How to deploy a model](docs/deploy/README.md)


## Acknowledgements

This code is based on work conducted in the following study:

*Identifying Animal Species in Camera Trap Images using Deep Learning and Citizen Science, 2018, Methods in Ecology and Evolution*

Authors: Marco Willi, Ross Tyzack Pitman, Anabelle W. Cardoso, Christina Locke, Alexandra Swanson, Amy Boyer, Marten Veldthuis, Lucy Fortson

Please cite as:
```
Willi M, Pitman RT, Cardoso AW, et al.
Identifying animal species in camera trap images using deep learning and citizen science.
Methods Ecol Evol. 2018;00:1–12. https://doi.org/10.1111/2041-210X.13099
```

The camera-trap images (330x330 pixels only) used in the study can be downloaded here:
https://doi.org/10.13020/D6T11K

The ResNet models are based on the implementation provided here:
https://github.com/raghakot/keras-resnet
