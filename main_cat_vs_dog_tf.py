""" Train a Model for Cats vs Dogs """
from data_processing.data_inventory import DatasetInventory
#from data_processing.data_reader import DatasetReader
from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
from data_processing.data_writer import DatasetWriter
from data_processing.tfr_splitter import TFRecordSplitter
from pre_processing.image_transformations import (
        preprocess_image,
        preprocess_image_default, resize_jpeg, resize_image)
import tensorflow as tf
import numpy as np
from data_processing.utils  import calc_n_batches_per_epoch, create_default_class_mapper
from config.config import logging
import matplotlib.pyplot as plt
from training.utils import LearningRateSetter, EarlyStopping, ReduceLearningRateOnPlateau

########################
# Parameters
#########################

path_to_images = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all"
path_to_tfr_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\4715\\"
path_to_model_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\models\\4715\\keras_tf\\"

path_to_images = '/host/data_hdd/images/4715/all'
path_to_tfr_output = '/host/data_hdd/camtrap/cats_vs_dogs/data'
path_to_model_output = '/host/data_hdd/camtrap/cats_vs_dogs/training/tf_keras/'



model_labels = ['primary']
label_mapper = None
n_classes = 2
batch_size = 128
image_save_side_max = 300
balanced_sampling_min=False
balanced_sampling_label_type = 'primary'
image_proc_args = {
    'output_height': 150,
    'output_width': 150,
    'image_means': [0, 0, 0],
    'image_stdevs': [1, 1, 1],
    'is_training': True,
    'resize_side_min': 224,
    'resize_side_max': 500}


####################################
# Convert some Parameters
####################################

balanced_sampling_label_type = 'labels/' + balanced_sampling_label_type

# Create Data Inventory
dataset_inventory = DatasetInventory()
dataset_inventory.create_from_class_directories(path_to_images)
dataset_inventory.remove_multi_label_records()


# Create TFRecod Encoder / Decoder
tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()


# Write TFRecord file from Data Inventory
tfr_writer = DatasetWriter(tfr_encoder_decoder.encode_record)
tfr_writer.encode_inventory_to_tfr(
        dataset_inventory,
        path_to_tfr_output + "all.tfrecord",
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"max_side": image_save_side_max},
        overwrite_existing_file=False)


# Split TFrecord into Train/Val/Test
tfr_splitter = TFRecordSplitter(
        files_to_split=path_to_tfr_output + "all.tfrecord",
        tfr_encoder=tfr_encoder_decoder.encode_record,
        tfr_decoder=tfr_encoder_decoder.decode_record)

tfr_splitter.split_tfr_file(output_path_main=path_to_tfr_output,
                            output_prefix="split",
                            split_names=['train', 'val', 'test'],
                            split_props=[0.9, 0.05, 0.05],
                            balanced_sampling_min=balanced_sampling_min,
                            balanced_sampling_label_type=balanced_sampling_label_type,
                            output_labels=model_labels,
                            overwrite_existing_files=False)


# Check numbers
tfr_splitter.log_record_numbers_per_file()
tfr_n_records = tfr_splitter.get_record_numbers_per_file()
tfr_splitter.label_to_numeric_mapper
num_to_label_mapper = {v:k for k, v in tfr_splitter.label_to_numeric_mapper['labels/primary'].items()}
tfr_splitter.get_record_numbers_per_file()

# Create Dataset Reader
data_reader = DatasetReader(tfr_encoder_decoder.decode_record)

# Calculate Dataset Image Means and Stdevs for a dummy batch
batch_data= data_reader.get_iterator(
        tfr_files=[tfr_splitter.get_split_paths()['train']],
        batch_size=1024,
        is_train=False,
        n_repeats=1,
        output_labels=model_labels,
        image_pre_processing_fun=preprocess_image_default,
        image_pre_processing_args=image_proc_args,
        max_multi_label_number=None,
        labels_are_numeric=True)


with tf.Session() as sess:
    data = sess.run(batch_data)


image_means = list(np.mean(data['images'], axis=(0, 1, 2)))
image_stdevs = list(np.std(data['images'], axis=(0, 1, 2)))

image_proc_args['image_means'] = image_means
image_proc_args['image_stdevs'] = image_stdevs


## plot some images and their labels to check
#for i in range(0, 30):
#    img = data['images'][i,:,:,:]
#    lbl = data['labels/primary'][i]
#    print("Label: %s" % num_to_label_mapper[int(lbl)])
#    plt.imshow(img)
#    plt.show()



# Prepare Data Feeders for Training / Validation Data
def input_feeder_train():
    batch_dict = data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['train']],
                batch_size=batch_size,
                is_train=True,
                n_repeats=1,
                output_labels=model_labels,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True,
                one_hot_labels=False,
                num_classes_list=[n_classes])

    features = {'images': batch_dict['images']}
    labels = {key: batch_dict[key] for key in batch_dict \
                 if key not in ['images', 'id']}
    return features, labels

def input_feeder_val():
    batch_dict = data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['val']],
                batch_size=batch_size,
                is_train=False,
                n_repeats=1,
                output_labels=model_labels,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True,
                one_hot_labels=False,
                num_classes_list=[n_classes])

    features = {'images': batch_dict['images']}
    labels = {key: batch_dict[key] for key in batch_dict \
                 if key not in ['images', 'id']}
    return features, labels


#test_data = input_feeder_val()
#with tf.Session() as sess:
#    imgs, labs = sess.run(test_data)
#
## plot some images and their labels to check
#for i in range(0, 33):
#    img = (imgs['images'][i,:,:,:] * image_proc_args['image_stdevs']) + image_proc_args['image_means']
#    lbl = labs['labels/primary'][i]
#    print("Label: %s" % num_to_label_mapper[int(lbl)])
#    plt.imshow(img)
#    plt.show()
#



n_batches_per_epoch_train = calc_n_batches_per_epoch(tfr_n_records['train'],
                                                     batch_size)

n_batches_per_epoch_val = calc_n_batches_per_epoch(tfr_n_records['val'],
                                                   batch_size)

######################################
# Model Training
######################################
from models.cats_vs_dogs import my_model_fn
from tensorflow.python.estimator.warm_starting_util import WarmStartSettings
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.contrib.learn import Experiment


hparams = {
    'momentum': 0.9,
    'learning_rate': 1e-2,
    'weight_decay': 1e-4,
    'image_target_size': [image_proc_args['output_height'], image_proc_args['output_width']],
    'image_n_color_channels': 3,
    'n_classes': [n_classes],
    'output_labels': ['labels/' + x for x in model_labels],
    'batch_size': batch_size,
    'transfer_learning': False,
    'transfer_learning_layers': 'dense',
    'inter_op_parallelism_threads': 0,
    'intra_op_parallelism_threads': 0,
    'multi_gpu': False
}


session_config = tf.ConfigProto(
  inter_op_parallelism_threads=hparams['inter_op_parallelism_threads'],
  intra_op_parallelism_threads=hparams['intra_op_parallelism_threads'],
  allow_soft_placement=True)


run_config = tf.estimator.RunConfig(
    model_dir=path_to_model_output,
    save_summary_steps=n_batches_per_epoch_train,
    save_checkpoints_steps=n_batches_per_epoch_train,
    session_config=session_config)


estimator = Estimator(
    model_fn=my_model_fn,
    params=hparams,
    config=run_config,
    warm_start_from=None
    )


early_stopping = EarlyStopping(stop_after_n_rounds=5, minimize=True)
reduce_lr_on_plateau = ReduceLearningRateOnPlateau(
        initial_lr=hparams['learning_rate'],
        reduce_after_n_rounds=3,
        patience_after_reduction=2,
        reduction_mult=0.1,
        min_lr=1e-5,
        minimize=True
        )

lr_setter = LearningRateSetter(reduce_lr_on_plateau.initial_lr)


# Train Model
while not early_stopping.stop_training:

    # Train model
    estimator.train(input_feeder_train, hooks=[lr_setter])

    # Eval Model
    res_val = estimator.evaluate(input_feeder_val)

    # add loss to early stopper
    loss_val = [res_val['loss/labels/' + x] for x in model_labels]
    early_stopping.addResult(loss_val[0])

    # Redue LR
    reduce_lr_on_plateau.addResult(loss_val[0])
    lr_setter.lr = reduce_lr_on_plateau.current_lr


predictor = estimator.predict(input_feeder_val)

pred_labels = ['labels/' + x for x in model_labels]

for pred in predictor:
    for pred_label in pred_labels:
        print(pred[(pred_label, 'probabilities')])
        print(num_to_label_mapper[pred[(pred_label, 'classes')]])
