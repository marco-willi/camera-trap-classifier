""" Class To Read TFRecord Files """
from config.config import logging
import tensorflow as tf


class DatasetReader(object):
    def __init__(self, tfr_decoder):
        self.tfr_decoder = tfr_decoder

    def _decode_tfrecord(self, **kwargs):
        """ Read TFRecord and return dictionary """
        return self.tfr_decoder(**kwargs)

    def set_decoder(self, tfr_decoder):
        """ Set Decoder """
        self.tfr_decoder = tfr_decoder

    def get_iterator(self, tfr_files, batch_size, is_train, n_repeats,
                     output_labels, max_multi_label_number=None,
                     buffer_size=2048, **kwargs):


                     # image_pre_processing_fun=None,
                     # image_pre_processing_args=None,
                     # buffer_size=2048,
                     # max_multi_label_number=None,
                     # choose_random_image=True,
                     # n_color_channels=3,
                     # decode_images=True):
        """ Create Iterator from TFRecord """

        assert type(output_labels) is list, "label_list must be of " + \
            " type list is of type %s" % type(output_labels)

        labels = ['labels/' + label for label in output_labels]

        dataset = tf.data.TFRecordDataset(tfr_files)

        # Encode TFrecord file
        # dataset = dataset.map(lambda x: self.tfr_decoder(
        #         serialized_example=x,
        #         output_labels=labels,
        #         n_color_channels=n_color_channels,
        #         choose_random_image=choose_random_image,
        #         image_pre_processing_fun=image_pre_processing_fun,
        #         image_pre_processing_args=image_pre_processing_args,
        #         decode_images=decode_images)
        #         )

        dataset = dataset.map(lambda x: self.tfr_decoder(
                serialized_example=x, output_labels=labels,
                 **kwargs)
                )

        if max_multi_label_number is not None:
            label_pad_dict = {x: [max_multi_label_number] for x in labels}
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=({'images': [None, None, None]},
                               label_pad_dict))

        else:
            dataset = dataset.batch(batch_size)

        # Return n_repeat batches of data
        # shuffle only for training
        if is_train:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.repeat(n_repeats)

        iterator = dataset.make_one_shot_iterator()

        # `features` is a dictionary in which each value is a batch of values
        # that feature; `labels` is a batch of labels.
        images, labels = iterator.get_next()

        return images, labels


class DatasetReaderMapper(object):
    def __init__(self, tfr_decoder):
        self.tfr_decoder = tfr_decoder

    def _decode_tfrecord(self, **kwargs):
        """ Read TFRecord and return dictionary """
        return self.tfr_decoder(**kwargs)

    def get_iterator(self, tfr_files, batch_size, is_train, n_repeats,
                     output_labels,
                     class_mapper,
                     image_pre_processing_fun=None,
                     image_pre_processing_args=None,
                     buffer_size=2048,
                     max_multi_label_number=None,
                     choose_random_image=True,
                     n_color_channels=3,
                     decode_images=True):
        """ Create Iterator from TFRecord """

        assert type(output_labels) is list, "label_list must be of " + \
            " type list is of type %s" % type(output_labels)

        labels = ['labels/' + label for label in output_labels]

        dataset = tf.data.TFRecordDataset(tfr_files)

        # Create Class Mappers
        def hash_mapper(output_label):
            if output_label not in class_mapper:
                logging.error("Label_Type %s not found in mapper"
                              % output_label)
            keys = class_mapper[output_label]['keys']
            values = class_mapper[output_label]['values']
            keys_tf = tf.cast(keys, tf.string)
            values_tf = tf.cast(values, tf.float32)
            table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(
                    keys_tf, values_tf), -1)
            return table

        mappers = dict()
        for ol, long_label in zip(output_labels, labels):
            mappers[long_label] = hash_mapper(ol)

        # Encode TFrecord file
        dataset = dataset.map(lambda x: self._decode_tfrecord(
                serialized_example=x,
                output_labels=labels,
                class_mappers=mappers,
                n_color_channels=n_color_channels,
                choose_random_image=choose_random_image,
                image_pre_processing_fun=image_pre_processing_fun,
                image_pre_processing_args=image_pre_processing_args,
                decode_images=decode_images)
                )

        if max_multi_label_number is not None:
            label_pad_dict = {x: [max_multi_label_number] for x in labels}
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=({'images': [None, None, None]},
                               label_pad_dict))

        else:
            dataset = dataset.batch(batch_size)

        # Return n_repeat batches of data
        # shuffle only for training
        if is_train:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.repeat(n_repeats)

        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer

        # `features` is a dictionary in which each value is a batch of values
        # that feature; `labels` is a batch of labels.
        images, labels = iterator.get_next()

        return images, labels, init_op
