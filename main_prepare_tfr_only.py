""" Train a Keras TF Model"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
#import matplotlib.pyplot as plt

from config.config import logging
from config.config import cfg
from training.configuration_data import get_label_info
from training.utils import (
        ReduceLearningRateOnPlateau, EarlyStopping, CSVLogger,
        ModelCheckpointer)
from training.model_library import create_model

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

# Convert label types to tfrecord compatible names (clean)
if cfg.current_exp['balanced_sampling_label_type'] is not None:
    cfg.current_exp['balanced_sampling_label_type'] = \
        'labels/' + cfg.current_exp['balanced_sampling_label_type']

label_types_to_model_clean = ['labels/' + x for x in
                              cfg.current_exp['label_types_to_model']]

# Create TFRecod Encoder / Decoder
logging.info("Creating TFRecord Data")
tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()

# Write TFRecord file from Data Inventory
tfr_writer = DatasetWriter(tfr_encoder_decoder.encode_record)
tfr_writer.encode_inventory_to_tfr(
        dataset_inventory,
        cfg.current_paths['tfr_master'],
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"max_side":
                                   cfg.current_exp['image_save_side_max']},
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
