""" Class To Read TFRecord Files """
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

        dataset = tf.data.TFRecordDataset(tfr_files)

        dataset = dataset.prefetch(buffer_size=buffer_size*10)

        # shuffle records only for training
        if is_train:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.apply(
              tf.contrib.data.map_and_batch(
                  lambda x: self.tfr_decoder(
                          serialized_example=x,
                          output_labels=output_labels,
                          **kwargs),
                  batch_size=batch_size,
                  num_parallel_batches=num_parallel_calls,
                  drop_remainder=drop_batch_remainder))

        dataset = dataset.repeat(n_repeats)

        dataset = dataset.prefetch(5)

        return dataset
