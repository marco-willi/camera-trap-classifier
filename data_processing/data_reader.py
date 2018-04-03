""" Class To Read TFRecord Files """
import tensorflow as tf

from config.config import logging


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

        dataset = dataset.prefetch(buffer_size=buffer_size*10)

        # shuffle records only for training
        if is_train:
            dataset = dataset.shuffle(buffer_size=buffer_size)

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

        dataset = dataset.prefetch(50)

        iterator = dataset.make_one_shot_iterator()

        batch = iterator.get_next()

        return batch
