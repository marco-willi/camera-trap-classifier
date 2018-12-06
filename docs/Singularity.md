# Singularity containers

For security reasons some environments do not allow the use of Docker containers. Instead, Singularity may be an option. Below an example of how we used Singularity on a super computing environemnt.

For more information: https://singularity.lbl.gov

```
# pull the tensorflow gpu image
singularity pull docker://tensorflow/tensorflow:1.12.0-gpu-py3

# since we are pulling the tensorflow image the camera trap classifier code is
# not included, we can download it outside the container
CODE=${HOME}/code
mkdir -p $CODE
cd $CODE
git clone https://github.com/marco-willi/camera-trap-classifier.git

# Start the container with the shell and map directories
# we assume that $HOME in the container is mapped to $HOME outside
singularity shell --nv -B /data/my_files/:/data/my_files/ ./tensorflow-1.12.0-gpu-py3.simg

# install missing packages
CODE=${HOME}/code
cd $CODE
pip install --user -e camera-trap-classifier

# Prepare paths
SAVE_ROOT_PATH=$HOME
TFR_FILES=/data/tfr_files/
RUN_OUTPUT=${SAVE_ROOT_PATH}/data/run_outputs/
MODEL_SAVE=${SAVE_ROOT_PATH}/data/model_save/
LABEL_MAPPINGS=/data/tfr_files/label_mapping.json

# create output paths if they dont exist
mkdir -p $RUN_OUTPUT
mkdir -p $MODEL_SAVE


# we can the run code with this (example for training a model)
cd ${CODE}/camera-trap-classifier

python -m camera_trap_classifier.train \
-train_tfr_path $TFR_FILES \
-val_tfr_path $TFR_FILES \
-test_tfr_path $TFR_FILES \
-class_mapping_json  $LABEL_MAPPINGS \
-run_outputs_dir $RUN_OUTPUT \
-model_save_dir $MODEL_SAVE \
-model ResNet18 \
-labels species count standing resting moving eating interacting babies \
-labels_loss_weights 1 0.2 0.2 0.2 0.2 0.2 0.2 0.2  \
-batch_size 256 \
-n_cpus 18 \
-n_gpus 2 \
-buffer_size 512 \
-max_epochs 70 \
-color_augmentation full_randomized \
-ignore_aspect_ratio
```

Other Commands:
```
# List running singularity containers
singularity instance.list
```
