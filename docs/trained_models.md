# Trained Models

The following trained models are available for downloading. The models were trained on AWS using a p3.2xlarge EC2  instance and Docker as described here [AWS example](../docs/Docker_GPU.md). The models can be apply as described here [Model Application](../README.md#5-model-use).

## Empty or Not Model

Download from: https://s3.msi.umn.edu/snapshotsafari/models/empty_or_not/ResNet18_448_v1.zip

* Number of capture events in training set: 3.66 million
* Architecture: ResNet18
* Input Resolution: 448x448
* Images: mainly from Tanzania (Serengeti National Park) and South Africa
* Output: is_blank - binary label indicating whether the capture is empty ('blank'), i.e. no animal or human


### Training Parameters

The model was trained using the following command:

```
sudo docker exec ctc ctc.train \
  -train_tfr_path $TFR_FILES \
  -val_tfr_path $TFR_FILES \
  -test_tfr_path $TFR_FILES \
  -class_mapping_json  $LABEL_MAPPINGS \
  -run_outputs_dir $RUN_OUTPUT \
  -model_save_dir $MODEL_SAVE \
  -model ResNet18 \
  -labels is_blank \
  -batch_size 64 \
  -n_cpus 8 \
  -n_gpus 1 \
  -buffer_size 512 \
  -max_epochs 70 \
  -reduce_lr_on_plateau_patience 4 \
  -early_stopping_patience 6 \
  -output_width 448 \
  -output_height 448 \
  -n_batches_per_epoch_train 5000 \
  -rotate_by_angle 0 \
  -zoom_factor 0
```

## Species Model

Download from: https://s3.msi.umn.edu/snapshotsafari/models/species/Xception_v1.zip

* Number of captures in training set: 1.84 million
* Architecture: Xception
* Input Resolution: 299x299
* Images: mainly from Tanzania (Serengeti National Part) and South Africa
* Output: species, counts, standing, resting, moving, eating, interacting, young_present
* Number of species: 85

<img src="https://github.com/marco-willi/camera-trap-classifier/blob/add_models/docs/figures/zebra_example.png"/>

*This figure shows a capture of a Zebra (see corresponding model output below).*

Example Model Output for the image above: [Xception_v1 Output](../docs/figures/example_pred_Xception_v1.json). This file also shows the complete list of species and all the other labels that the model was trained to classify.

### Training Parameters

The model was trained using the following command:

```
sudo docker exec ctc ctc.train \
  -train_tfr_path $TFR_FILES \
  -val_tfr_path $TFR_FILES \
  -test_tfr_path $TFR_FILES \
  -class_mapping_json  $LABEL_MAPPINGS \
  -run_outputs_dir $RUN_OUTPUT \
  -model_save_dir $MODEL_SAVE \
  -model Xception \
  -labels species count standing resting moving eating interacting young_present \
  -batch_size 32 \
  -n_cpus 8 \
  -n_gpus 1 \
  -buffer_size 512 \
  -max_epochs 70 \
  -reduce_lr_on_plateau_patience 6 \
  -early_stopping_patience 10 \
  -rotate_by_angle 0 \
  -zoom_factor 0 \
  -n_batches_per_epoch_train 10000 \
  -max_epochs 70
```

## Evaluation Results

### Results on Test-Data (in sample data)

The following report shows evaluation results on the test data: [Evaluation Results](../docs/figures/Evaluation_SnapshotSafariModels.pdf).

Shown are results separated by different locations / parks. The overwhelming majority of the images is from 'SER' the Serengeti national park.

The data is refered to as 'in sample data' because it is based on data from the same locations as used in training the model. This may introduce a positive bias to the results. Furthermore, the species model is evaluated on species images only (i.e. the model could not make the error of predicting a species when there is none).


### Results on Independent Data (out of sample data)

The following reports show model evaluations using independent test data:

* Grumeti: [Evaluation Results Grumeti](../docs/figures/GRU_S1_model_evaluation.pdf)
* Enonkishu: [Evaluation Results Enonkishu](../docs/figures/ENO_S1_model_evaluation.pdf)
* Kruger: [Evaluation Results Kruger](../docs/figures/KRU_S1_model_evaluation.pdf)

The evaluation from these locations is based on a realistic scenario, on data collected from locations (Enonkishu and Kruger) the model has not seen or mostly not seen (Grumeti), as well as by applying a two-stage process: 1) identify images with animals and 2) identify species on images considered as containing an animal. Note that some (rare) species from Kruger / Enonkishu are completely unknown to the model and thus not predictable.
