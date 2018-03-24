""" Train Model for CamCat """
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
from data_processing.utils import calc_n_batches_per_epoch
from config.config import logging
#import matplotlib.pyplot as plt
from training.utils import (
    LearningRateSetter, EarlyStopping, ReduceLearningRateOnPlateau, CSVLogger)

########################
# Parameters
#########################

path_to_images = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\ss"
path_to_tfr_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\ss\\"
path_to_model_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\models\\ss\\resnet_keras_tf\\"

path_to_images = '/host/data_hdd/images/camera_catalogue/all'
path_to_model_output = '/host/data_hdd/camtrap/camera_catalogue/training/tf_keras_one_hot/'
path_to_tfr_output = '/host/data_hdd/camtrap/camera_catalogue/data/'

labels_all = {
  'primary': [
    'bat', 'hartebeest', 'insect', 'klipspringer', 'hyaenabrown',
    'domesticanimal', 'otter', 'hyaenaspotted', 'MACAQUE', 'aardvark',
    'reedbuck', 'waterbuck', 'bird', 'genet', 'blank', 'porcupine',
    'caracal', 'aardwolf', 'bushbaby', 'bushbuck', 'mongoose', 'polecat',
    'honeyBadger', 'reptile', 'cheetah', 'pangolin', 'giraffe', 'rodent',
    'leopard', 'roansable', 'hippopotamus', 'rabbithare', 'warthog', 'kudu',
    'batEaredFox', 'gemsbock', 'africancivet', 'rhino', 'wildebeest',
    'monkeybaboon', 'zebra', 'bushpig', 'elephant', 'nyala', 'jackal',
    'serval', 'buffalo', 'vehicle', 'eland', 'impala', 'lion',
    'wilddog', 'duikersteenbok', 'HUMAN', 'wildcat']}

keep_labels_all = {
  'primary': [
    'bat', 'hartebeest', 'insect', 'klipspringer', 'hyaenabrown',
    'domesticanimal', 'hyaenaspotted', 'aardvark',
    'reedbuck', 'waterbuck', 'bird', 'genet', 'blank', 'porcupine',
    'caracal', 'aardwolf', 'bushbaby', 'bushbuck', 'mongoose',
    'honeyBadger', 'cheetah', 'giraffe', 'rodent',
    'leopard', 'roansable', 'hippopotamus', 'rabbithare', 'warthog', 'kudu',
    'batEaredFox', 'gemsbock', 'africancivet', 'rhino', 'wildebeest',
    'monkeybaboon', 'zebra', 'bushpig', 'elephant', 'nyala', 'jackal',
    'serval', 'buffalo', 'vehicle', 'eland', 'impala', 'lion',
    'wilddog', 'duikersteenbok', 'HUMAN', 'wildcat']}


keep_labels_species = {x: y.copy() for x, y in keep_labels_all.items()}
keep_labels_species['primary'].remove('vehicle')
keep_labels_species['primary'].remove('blank')

map_labels_empty = {'primary': {x: 'species' for x in keep_labels_all['primary'] if x not in ['vehicle', 'blank']}}
map_labels_empty['primary']['vehicle'] = 'vehicle'
map_labels_empty['primary']['blank'] = 'blank'

label_types_to_model = ['primary']
keep_only_labels=keep_labels_species
class_mapping=None
batch_size = 128
image_save_side_max = 330
balanced_sampling_min= False
balanced_sampling_label_type = 'primary'
image_proc_args = {
    'output_height': 224,
    'output_width': 224,
    'image_means': [0, 0, 0],
    'image_stdevs': [1, 1, 1],
    'is_training': True,
    'resize_side_min': 224,
    'resize_side_max': 500}


####################################
# Convert some Parameters
####################################

if balanced_sampling_label_type is not None:
    balanced_sampling_label_type = 'labels/' + balanced_sampling_label_type

label_types_to_model_clean = ['labels/' + x for x in label_types_to_model]


# Create Data Inventory
dataset_inventory = DatasetInventory()
dataset_inventory.create_from_class_directories(path_to_images)
dataset_inventory.label_handler.remove_multi_label_records()
dataset_inventory.log_stats()

# Create TFRecod Encoder / Decoder
tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()

# Write TFRecord file from Data Inventory
tfr_writer = DatasetWriter(tfr_encoder_decoder.encode_record)
tfr_writer.encode_inventory_to_tfr(
        dataset_inventory,
        path_to_tfr_output + "all.tfrecord",
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"max_side": image_save_side_max},
        overwrite_existing_file=False,
        prefix_to_labels='labels/')

# Split TFrecord into Train/Val/Test
logging.debug("Creating TFRecordSplitter")
tfr_splitter = TFRecordSplitter(
        files_to_split=path_to_tfr_output + "all.tfrecord",
        tfr_encoder=tfr_encoder_decoder.encode_record,
        tfr_decoder=tfr_encoder_decoder.decode_record)

logging.debug("Splitting TFR File")
tfr_splitter.split_tfr_file(output_path_main=path_to_tfr_output,
                            output_prefix="split",
                            split_names=['train', 'val', 'test'],
                            split_props=[0.9, 0.05, 0.05],
                            balanced_sampling_min=balanced_sampling_min,
                            balanced_sampling_label_type=balanced_sampling_label_type,
                            output_labels=label_types_to_model,
                            overwrite_existing_files=False,
                            keep_only_labels=keep_only_labels,
                            class_mapping=class_mapping)


# Check numbers
tfr_splitter.log_record_numbers_per_file()
tfr_n_records = tfr_splitter.get_record_numbers_per_file()
tfr_splitter.label_to_numeric_mapper
num_to_label_mapper = {
    k: {v2: k2 for k2, v2 in v.items()}
    for k, v in tfr_splitter.label_to_numeric_mapper.items()}
n_classes_per_label_type = [len(num_to_label_mapper[x]) for x in \
                            label_types_to_model_clean]

tfr_splitter.get_record_numbers_per_file()

# Create Dataset Reader
logging.debug("Create Dataset Reader")
data_reader = DatasetReader(tfr_encoder_decoder.decode_record)

# Calculate Dataset Image Means and Stdevs for a dummy batch
logging.debug("Get Dataset Reader for calculating datset stats")
batch_data = data_reader.get_iterator(
        tfr_files=[tfr_splitter.get_split_paths()['train']],
        batch_size=1024,
        is_train=False,
        n_repeats=1,
        output_labels=label_types_to_model,
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

logging.info("Image Means: %s" % image_means)
logging.info("Image Stdevs: %s" % image_stdevs)

# # plot some images and their labels to check
# for i in range(0, 30):
#     img = data['images'][i,:,:,:]
#     lbl = data['labels/primary'][i]
#     print("Label: %s" % num_to_label_mapper[int(lbl)])
#     plt.imshow(img)
#     plt.show()


# Prepare Data Feeders for Training / Validation Data
def input_feeder_train():
    batch_dict = data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['train']],
                batch_size=batch_size,
                is_train=True,
                n_repeats=None,
                output_labels=label_types_to_model,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True,
                one_hot_labels=False,
                num_classes_list=n_classes_per_label_type)

    features = {'images': batch_dict['images']}
    labels = {key: batch_dict[key] for key in batch_dict \
                 if key not in ['images', 'id']}
    return features, labels

def input_feeder_val():
    batch_dict = data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['val']],
                batch_size=batch_size,
                is_train=False,
                n_repeats=None,
                output_labels=label_types_to_model,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True,
                one_hot_labels=False,
                num_classes_list=n_classes_per_label_type)

    features = {'images': batch_dict['images']}
    labels = {key: batch_dict[key] for key in batch_dict \
                 if key not in ['images', 'id']}
    return features, labels


def input_feeder_test():
    batch_dict = data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['test']],
                batch_size=batch_size,
                is_train=False,
                n_repeats=1,
                output_labels=label_types_to_model,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True,
                one_hot_labels=False,
                num_classes_list=n_classes_per_label_type)

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


from models.tf_keras_resnet import my_model_fn
from tensorflow.python.estimator.warm_starting_util import WarmStartSettings
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.contrib.learn import Experiment

hparams = {
    'momentum': 0.9,
    'learning_rate': 1e-2,
    'weight_decay': 1e-4,
    'image_target_size': [image_proc_args['output_height'], image_proc_args['output_width']],
    'image_n_color_channels': 3,
    'n_classes': n_classes_per_label_type,
    'output_labels': label_types_to_model_clean,
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
        minimize=True)


lr_setter = LearningRateSetter(reduce_lr_on_plateau.initial_lr)


logger = CSVLogger(path_to_model_output + 'log.csv',
                   metrics_names=['val_loss_' + x for x in label_types_to_model] + \
                   ['val_accuracy_' + x for x in label_types_to_model])

# Train Model
epoch = 0
logging.debug("Start Model Training")
while not early_stopping.stop_training:

    # Train model
    estimator.train(input_feeder_train, hooks=[lr_setter], steps=n_batches_per_epoch_train)

    # Eval Model
    res_val = estimator.evaluate(input_feeder_val, steps=n_batches_per_epoch_val)

    logging.info("Eval Results")
    for metric, value in res_val.items():
        logging.info("    Metric: %s Res %s" % (metric, value))

    # add loss to early stopper
    loss_val = [res_val['loss/labels/' + x] for x in label_types_to_model]
    acc_val = [res_val['accuracy/labels/' + x] for x in label_types_to_model]

    loss_total = res_val['loss']
    early_stopping.addResult(loss_total)

    # Redue LR
    reduce_lr_on_plateau.addResult(loss_total)
    lr_setter.lr = reduce_lr_on_plateau.current_lr

    # add result to log file
    vals_to_log = loss_val + acc_val
    logger.addResults(epoch, vals_to_log)
    epoch += 1


predictor = estimator.predict(input_feeder_test)

pred_labels = label_types_to_model_clean
logging.debug("Start Predictions")
for pred in predictor:
    for pred_label in pred_labels:
        logging.info(pred[(pred_label, 'probabilities')])
        logging.info(print(num_to_label_mapper[pred_label][pred[(pred_label, 'classes')]]))
