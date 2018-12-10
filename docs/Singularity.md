# Singularity containers

For security reasons some environments do not allow the use of Docker containers. Instead, Singularity may be an option. Below an example of how we used Singularity on a super computing environemnt.

For more information: https://singularity.lbl.gov

```
# pull the docker image from Dockerhub (note: no guarantee this is up-to-date)
singularity pull docker://will5448/camera-trap-classifier:latest-gpu

# run a program
# map important paths with -B option
singularity exec --nv -B /data/my_files/:/data/my_files/ ./camera-trap-classifier-latest-gpu.simg echo "TEST"

# Alternatively switch into the singularit container and run commands there
singularity run ./camera-trap-classifier-latest-gpu.simg
```


```
# Example Usage

cd $HOME

# Prepare paths
SAVE_ROOT_PATH=$HOME
TFR_FILES=/data/tfr_files/
RUN_OUTPUT=${SAVE_ROOT_PATH}/data/run_outputs/
MODEL_SAVE=${SAVE_ROOT_PATH}/data/model_save/
LABEL_MAPPINGS=/data/tfr_files/label_mapping.json

# create output paths if they dont exist
mkdir -p $RUN_OUTPUT
mkdir -p $MODEL_SAVE

singularity exec --nv -B /data/my_files/:/data/my_files/ ./camera-trap-classifier-latest-gpu.simg \
  ctc.train \
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
  -max_epochs 70
```
