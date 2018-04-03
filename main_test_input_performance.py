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
logging.info("Creating TFRecordSplitter")
tfr_splitter = TFRecordSplitter(
        files_to_split=cfg.current_paths['tfr_master'],
        tfr_encoder=tfr_encoder_decoder.encode_record,
        tfr_decoder=tfr_encoder_decoder.decode_record)

#
# class DatasetReader2(object):
#     def __init__(self, tfr_decoder):
#         self.tfr_decoder = tfr_decoder
#     def get_iterator(self, tfr_files, batch_size, is_train, n_repeats,
#                      output_labels, max_multi_label_number=None,
#                      buffer_size=10192, num_parallel_calls=4,
#                      drop_batch_remainder=True, **kwargs):
#         """ Create Iterator from TFRecord """
#         assert type(output_labels) is list, "label_list must be of " + \
#             " type list is of type %s" % type(output_labels)
#         labels = ['labels/' + label for label in output_labels]
#         logging.info("Creating dataset TFR iterator")
#         dataset = tf.data.TFRecordDataset(tfr_files)
#         # dataset = dataset.apply(tf.contrib.data.map_and_batch(
#         #                 map_func=lambda x: self.tfr_decoder(
#         #                         serialized_example=x,
#         #                         output_labels=labels,
#         #                         **kwargs), batch_size=batch_size))
#         dataset = dataset.map(lambda x: self.tfr_decoder(
#                 serialized_example=x,
#                 output_labels=labels,
#                 **kwargs), num_parallel_calls=num_parallel_calls
#                 )
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.repeat(n_repeats)
#         dataset = dataset.prefetch(1)
#         #iterator = dataset.make_one_shot_iterator()
#         iterator = dataset.make_initializable_iterator()
#         return iterator
#         # batch = iterator.get_next()
#         # return batch

class DatasetReader(object):
    def __init__(self, tfr_decoder):
        self.tfr_decoder = tfr_decoder
    def get_iterator(self, tfr_files, batch_size, is_train, n_repeats,
                     output_labels, max_multi_label_number=None,
                     buffer_size=10192, num_parallel_calls=4,
                     drop_batch_remainder=True, **kwargs):
        """ Create Iterator from TFRecord """
        assert type(output_labels) is list, "label_list must be of " + \
            " type list is of type %s" % type(output_labels)
        labels = ['labels/' + label for label in output_labels]
        logging.info("Creating dataset TFR iterator")
        dataset = tf.data.TFRecordDataset(tfr_files)
        # shuffle records only for training
        if is_train:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.prefetch(buffer_size=batch_size*10)
        dataset = dataset.map(lambda x: self.tfr_decoder(
                serialized_example=x,
                output_labels=labels,
                **kwargs), num_parallel_calls=num_parallel_calls
                )
        if max_multi_label_number is not None:
            label_pad_dict = {x: [max_multi_label_number] for x in labels}
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=({'images': [None, None, None],
                                'id': [None],
                               **label_pad_dict}))
        else:
            if drop_batch_remainder:
                dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            else:
                dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(n_repeats)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        return batch


dataset_reader = DatasetReader(tfr_encoder_decoder.decode_record)
iterator = data_reader.get_iterator(
        tfr_files=[tfr_splitter.get_split_paths()['train']],
        batch_size=cfg.cfg['general']['batch_size'],
        is_train=True,
        n_repeats=None,
        output_labels=cfg.current_exp['label_types_to_model'],
        image_pre_processing_fun=preprocess_image,
        image_pre_processing_args={**cfg.current_exp['image_processing'],
                                   'is_training': True},
        max_multi_label_number=None,
        buffer_size=cfg.cfg['general']['buffer_size'],
        num_parallel_calls=cfg.cfg['general']['number_of_cpus'],
        labels_are_numeric=True)


# 73 seconds buffer 20480, prefetch 1, batch 2048, ncpus 32
# 74 seconds buffer 20480*10, prefetch 1, batch 2048, ncpus 32
# 76 seconds buffer 20480, prefetch 1, batch 2048, ncpus 8
# 88 seconds buffer 20480, prefetch 1, batch 2048, ncpus 32

import time
with tf.Session() as sess:
    # sess.run(init_op)
    # sess.run(iterator.initializer)
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    n_iter = 0
    n_records = 0
    logging.info("Starting the iteration")
    t_start_all = time.time()
    while True:
        t_start = time.time()
        try:
            batch_data = sess.run(iterator)
            n_records += batch_data['id'].shape[0]
            logging.info("Took %f seconds for iteraton %s" % (time.time()-t_start, str(n_iter)))
            logging.info("----Records processed: %s" % str(n_records))
            n_iter += 1
        except tf.errors.OutOfRangeError:
            logging.info("Out of Range Error")
            break
    # coord.request_stop()
    # coord.join(threads)
    logging.info("End of Session - total time: %s seconds" % (time.time()-t_start_all))

logging.info("Finished reading: %s" % cfg.current_paths['tfr_master'])



record_iterator = tf.python_io.tf_record_iterator(path=cfg.current_paths['tfr_master'])
