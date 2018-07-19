""" Class To Read TFRecord Files
    https://www.tensorflow.org/performance/datasets_performance
"""
import tensorflow as tf
import logging


logger = logging.getLogger(__name__)


class DatasetReader(object):
    def __init__(self, tfr_decoder):
        self.tfr_decoder = tfr_decoder

    def get_iterator(self, tfr_files, batch_size, is_train, n_repeats,
                     output_labels,
                     buffer_size=10192, num_parallel_calls=4,
                     drop_batch_remainder=True, **kwargs):
        """ Create Iterator from TFRecord """

        assert type(output_labels) is list, "label_list must be of " + \
            " type list is of type %s" % type(output_labels)

        logger.info("Creating dataset TFR iterator")

        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        # dataset = tf.data.TFRecordDataset(tfr_files)

        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=4))

        dataset = dataset.prefetch(buffer_size=batch_size)

        # shuffle records only for training
        if is_train:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=buffer_size,
                    count=n_repeats))

        dataset = dataset.apply(
              tf.contrib.data.map_and_batch(
                  lambda x: self.tfr_decoder(
                          serialized_example=x,
                          output_labels=output_labels,
                          **kwargs),
                  batch_size=batch_size,
                  drop_remainder=drop_batch_remainder))

        if not is_train:
            dataset = dataset.repeat(n_repeats)

        return dataset
