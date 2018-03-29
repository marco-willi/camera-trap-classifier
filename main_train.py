""" Train a Keras TF Model"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import SGD, Adagrad, RMSprop
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, TensorBoard)
from tensorflow.python.keras import backend as K
# import matplotlib.pyplot as plt

from config.config import logging
from config.config import cfg
from training.configuration_data import get_label_info
from training.utils import (
        ReduceLearningRateOnPlateau, EarlyStopping, CSVLogger)
from models.resnet_keras_mod import ResnetBuilder

from data_processing.data_inventory import DatasetInventory
from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
from data_processing.data_writer import DatasetWriter
from data_processing.tfr_splitter import TFRecordSplitter
from pre_processing.image_transformations import (
        preprocess_image,
        preprocess_image_default, resize_jpeg, resize_image)
from data_processing.utils import calc_n_batches_per_epoch

# get label information
logging.info("Getting Label Information")
labels_data = get_label_info(location=cfg.cfg['run']['location'],
                             experiment=cfg.cfg['run']['experiment'])


# Create Data Inventory
logging.info("Building Dataset Inventory")
dataset_inventory = DatasetInventory()
dataset_inventory.create_from_panthera_csv(cfg.current_paths['inventory'])
dataset_inventory.label_handler.remove_multi_label_records()
dataset_inventory.log_stats()


if cfg.current_exp['balanced_sampling_label_type'] is not None:
    cfg.current_exp['balanced_sampling_label_type'] = 'labels/' + cfg.current_exp['balanced_sampling_label_type']

label_types_to_model_clean = ['labels/' + x for x in cfg.current_exp['label_types_to_model']]

# Create TFRecod Encoder / Decoder
logging.info("Creating TFRecord Data")
tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()


# Write TFRecord file from Data Inventory
tfr_writer = DatasetWriter(tfr_encoder_decoder.encode_record)
tfr_writer.encode_inventory_to_tfr(
        dataset_inventory,
        cfg.current_paths['tfr_master'],
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"max_side": cfg.current_exp['image_save_side_max']},
        overwrite_existing_file=False,
        prefix_to_labels='labels/')

# Split TFrecord into Train/Val/Test
logging.debug("Creating TFRecordSplitter")
tfr_splitter = TFRecordSplitter(
        files_to_split=cfg.current_paths['tfr_master'],
        tfr_encoder=tfr_encoder_decoder.encode_record,
        tfr_decoder=tfr_encoder_decoder.decode_record)

split_names = [x for x in cfg.current_exp['training_splits']]
split_props = [cfg.current_exp['training_splits'][x] for x in split_names]

logging.debug("Splitting TFR File")
tfr_splitter.split_tfr_file(
    output_path_main=cfg.current_paths['exp_data'],
    #output_path_main='/host/data_hdd/southern_africa/experiments/species/data/',
    output_prefix="split",
    split_names=split_names,
    split_props=split_props,
    balanced_sampling_min=cfg.current_exp['balanced_sampling_min'],
    balanced_sampling_label_type=cfg.current_exp['balanced_sampling_label_type'],
    output_labels=cfg.current_exp['label_types_to_model'],
    overwrite_existing_files=False,
    keep_only_labels=labels_data['keep_labels'],
    class_mapping=labels_data['label_mapping'])

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

for label_type, labels in tfr_splitter.all_labels.items():
    for label, no_recs in labels.items():
        label_char = num_to_label_mapper[label_type][label]
        logging.info("Label Type: %s Label: %s Records: %s" %
                     (label_type, label_char, no_recs))

# Create Dataset Reader
logging.info("Create Dataset Reader")
data_reader = DatasetReader(tfr_encoder_decoder.decode_record)

# Calculate Dataset Image Means and Stdevs for a dummy batch
logging.info("Get Dataset Reader for calculating datset stats")
batch_data = data_reader.get_iterator(
        tfr_files=[tfr_splitter.get_split_paths()['train']],
        batch_size=1024,
        is_train=False,
        n_repeats=1,
        output_labels=cfg.current_exp['label_types_to_model'],
        image_pre_processing_fun=preprocess_image_default,
        image_pre_processing_args={**cfg.current_exp['image_processing'],
                                   'is_training': False},
        max_multi_label_number=None,
        labels_are_numeric=True)

logging.info("Calculating image means and stdevs")
with tf.Session() as sess:
    data = sess.run(batch_data)

image_means = list(np.mean(data['images'], axis=(0, 1, 2)))
image_stdevs = list(np.std(data['images'], axis=(0, 1, 2)))

cfg.current_exp['image_processing']['image_means'] = image_means
cfg.current_exp['image_processing']['image_stdevs'] = image_stdevs

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
logging.info("Preparing Data Feeders")
def input_feeder_train():
    return data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['train']],
                batch_size=cfg.current_model['batch_size'],
                is_train=True,
                n_repeats=None,
                output_labels=cfg.current_exp['label_types_to_model'],
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args={**cfg.current_exp['image_processing'],
                                           'is_training': True},
                max_multi_label_number=None,
                labels_are_numeric=True)

def input_feeder_val():
    return data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['validation']],
                batch_size=cfg.current_model['batch_size'],
                is_train=False,
                n_repeats=None,
                output_labels=cfg.current_exp['label_types_to_model'],
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args={**cfg.current_exp['image_processing'],
                                           'is_training': False},
                max_multi_label_number=None,
                labels_are_numeric=True)

def input_feeder_test():
    return data_reader.get_iterator(
                tfr_files=[tfr_splitter.get_split_paths()['test']],
                batch_size=cfg.current_model['batch_size'],
                is_train=False,
                n_repeats=None,
                output_labels=cfg.current_exp['label_types_to_model'],
                image_pre_processing_fun=preprocess_image_default,
                image_pre_processing_args={**cfg.current_exp['image_processing'],
                                           'is_training': False},
                max_multi_label_number=None,
                labels_are_numeric=True)

logging.info("Calculating batches per epoch")
n_batches_per_epoch_train = calc_n_batches_per_epoch(tfr_n_records['train'],
                                                     cfg.current_model['batch_size'])

n_batches_per_epoch_val = calc_n_batches_per_epoch(tfr_n_records['validation'],
                                                   cfg.current_model['batch_size'])

n_batches_per_epoch_val = calc_n_batches_per_epoch(tfr_n_records['test'],
                                                   cfg.current_model['batch_size'])

# Load Model Architecture and build output layer
logging.info("Building Model")

def create_model(input_feeder, target_labels):
    """ Create Keras Model """
    data = input_feeder()
    model_input = Input(tensor=data['images'])
    res_builder = ResnetBuilder()
    model_flat = res_builder.build_resnet_18(model_input)
    all_outputs = list()

    for n, name in zip(n_classes_per_label_type, target_labels):
        all_outputs.append(Dense(units=n, kernel_initializer="he_normal",
                           activation='softmax', name=name)(model_flat))

    model = Model(inputs=model_input, outputs=all_outputs)

    target_tensors = {x: tf.cast(data[x], tf.float32)
                      for x in target_labels}

    opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
    # opt =  RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', 'sparse_top_k_categorical_accuracy'],
                  target_tensors=target_tensors)
    return model


# Callbacks and Monitors
early_stopping = EarlyStopping(stop_after_n_rounds=7, minimize=True)
reduce_lr_on_plateau = ReduceLearningRateOnPlateau(
        reduce_after_n_rounds=3,
        patience_after_reduction=2,
        reduction_mult=0.1,
        min_lr=1e-5,
        minimize=True)

logger = CSVLogger(
    cfg.current_paths['run_data'] + 'training.log',
    metrics_names=['val_loss', 'val_acc',
                   'val_sparse_top_k_categorical_accuracy', 'learning_rate'])


checkpointer = ModelCheckpoint(
        filepath=cfg.current_paths['run_data'] + 'weights.{epoch:02d}-{loss:.2f}.hdf5',
        monitor='loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto', period=1)

tensorboard = TensorBoard(log_dir=cfg.current_paths['run_data'],
                          histogram_freq=0,
                          batch_size=cfg.current_model['batch_size'], write_graph=True,
                          write_grads=False, write_images=False)


train_model = create_model(input_feeder_train, label_types_to_model_clean)
val_model = create_model(input_feeder_val, label_types_to_model_clean)


for i in range(0, 70):
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

    vals_to_log = list()

    for metric, value in zip(val_model.metrics_names, results):

        logging.info("Eval - %s: %s" % (metric, value))
        vals_to_log.append(value)

    # Log Results on Validation Set
    vals_to_log.append(K.eval(train_model.optimizer.lr))

    logger.addResults(i+1, vals_to_log)

    # Reduce Learning Rate if necessary
    model_lr = K.eval(train_model.optimizer.lr)
    reduce_lr_on_plateau.addResult(val_loss, model_lr)
    if reduce_lr_on_plateau.reduced_in_last_step:
        K.set_value(train_model.optimizer.lr, reduce_lr_on_plateau.new_lr)
        logging.info("Setting LR to: %s" % K.eval(train_model.optimizer.lr))

    # Check if training should be stopped
    early_stopping.addResult(val_loss)
    if early_stopping.stop_training:
        logging.info("Early Stopping of Model Training after %s Epochs" %
                     (i+1))
        break

logging.info("Finished Model Training")
