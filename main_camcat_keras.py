""" Train a Model for CamCat """
from data_processing.data_inventory import DatasetInventory
from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
from data_processing.data_writer import DatasetWriter
from data_processing.tfr_splitter import TFRecordSplitter
from pre_processing.image_transformations import (
        preprocess_image,
        preprocess_image_default, resize_jpeg, resize_image)
import tensorflow as tf
import numpy as np
from data_processing.utils  import calc_n_batches_per_epoch
from config.config import logging
#import matplotlib.pyplot as plt

########################
# Parameters
#########################

path_to_images = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\ss"
path_to_tfr_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\ss\\"
path_to_model_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\models\\ss\\resnet_keras\\"

path_to_images = '/host/data_hdd/images/camera_catalogue/all'
path_to_model_output = '/host/data_hdd/camtrap/camera_catalogue/training/keras_only_blank/'
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
keep_only_labels=keep_labels_all
class_mapping=map_labels_empty
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

tfr_splitter.get_record_numbers_per_file()
tfr_splitter.all_labels
n_classes_per_label_type = [len(tfr_splitter.all_labels[x]) for x in \
                            label_types_to_model_clean]

logging.info("N_classes per label type: %s" % n_classes_per_label_type)

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


## plot some images and their labels to check
#for i in range(0, 30):
#    img = data['images'][i,:,:,:]
#    lbl = data['labels/primary'][i]
#    print("Label: %s" % num_to_label_mapper[int(lbl)])
#    plt.imshow(img)
#    plt.show()
#


# Prepare Data Feeders for Training / Validation Data
def input_feeder_train():
    return data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['train']],
                batch_size=batch_size,
                is_train=True,
                n_repeats=None,
                output_labels=label_types_to_model,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True)


def input_feeder_val():
    return data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['val']],
                batch_size=batch_size,
                is_train=False,
                n_repeats=None,
                output_labels=label_types_to_model,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True)

def input_feeder_test():
    return data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['test']],
                batch_size=batch_size,
                is_train=False,
                n_repeats=None,
                output_labels=label_types_to_model,
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args=image_proc_args,
                max_multi_label_number=None,
                labels_are_numeric=True)

n_batches_per_epoch_train = calc_n_batches_per_epoch(tfr_n_records['train'],
                                                     batch_size)

n_batches_per_epoch_val = calc_n_batches_per_epoch(tfr_n_records['val'],
                                                   batch_size)

# Define Model
from models.resnet_keras_mod import ResnetBuilder
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import SGD, Adagrad, RMSprop
from training.utils import ReduceLearningRateOnPlateau, EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.python.keras import backend as K


def create_model(input_feeder, target_labels):

    data = input_feeder()
    model_input = Input(tensor=data['images'])
    res_builder = ResnetBuilder()
    model_flat = res_builder.build_resnet_18(model_input)

    # TODO FIX:
    all_outputs = list()

    for n, name in zip(n_classes_per_label_type, target_labels):
        all_outputs.append(Dense(units=n, kernel_initializer="he_normal",
                           activation='softmax', name=name)(model_flat))

    model = Model(inputs=model_input, outputs=all_outputs)

    # TODO: build multiple outputs in architecture and map to labels
    target_tensors = {x: tf.cast(data[x], tf.float32) \
                      for x in target_labels}

    opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
    # opt =  RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', 'sparse_top_k_categorical_accuracy'],
                  target_tensors=target_tensors)
    return model


# Callbacks and Monitors
early_stopping = EarlyStopping(stop_after_n_rounds=6, minimize=True)
reduce_lr_on_plateau = ReduceLearningRateOnPlateau(
        initial_lr=0.01,
        reduce_after_n_rounds=3,
        patience_after_reduction=2,
        reduction_mult=0.1,
        min_lr=1e-5,
        minimize=True)

csv_logger = CSVLogger(path_to_model_output + 'training.log')

checkpointer = ModelCheckpoint(
        filepath=path_to_model_output + 'weights.{epoch:02d}-{loss:.2f}.hdf5',
        monitor='loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto', period=1)

tensorboard = TensorBoard(log_dir=path_to_model_output,
                          histogram_freq=0,
                          batch_size=batch_size, write_graph=True,
                          write_grads=False, write_images=False)


train_model = create_model(input_feeder_train, label_types_to_model_clean)
val_model = create_model(input_feeder_val, label_types_to_model_clean)


for i in range(0, 50):
    logging.info("Starting Epoch %s" % (i+1))
    train_model.fit(epochs=i+1,
                    steps_per_epoch=n_batches_per_epoch_train,
                    initial_epoch=i,
                    callbacks=[checkpointer, tensorboard])

    # Copy weights from training model to validation model
    weights = train_model.get_weights()
    val_model.set_weights(weights)

    # Run evaluation model
    results = val_model.evaluate(steps=n_batches_per_epoch_val)

    val_loss = results[val_model.metrics_names == 'loss']

    for val, metric in zip(val_model.metrics_names, results):

        logging.info("Eval - %s: %s" % (metric, val))

    # Reduce Learning Rate if necessary
    reduce_lr_on_plateau.current_lr = K.eval(train_model.optimizer.lr)
    reduce_lr_on_plateau.addResult(val_loss)
    if reduce_lr_on_plateau.reduced_in_last_step:
        train_model.optimizer.lr.assign(reduce_lr_on_plateau.current_lr)
        logging.info("Setting LR to: %s" % reduce_lr_on_plateau.current_lr)

    # Check if training should be stopped
    early_stopping.addResult(val_loss)
    if early_stopping.stop_training:
        logging.info("Early Stopping of Model Training after %s Epochs" %
                     (i+1))
        break
