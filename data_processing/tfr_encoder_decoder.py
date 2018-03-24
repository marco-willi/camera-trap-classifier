""" Class To Encode and Decode TFRecords"""
import tensorflow as tf
from data_processing.utils import (
        wrap_int64, wrap_bytes, wrap_dict_bytes_list,
        wrap_dict_bytes_str,
        _bytes_feature_list,
        wrap_dict_int64_list)


class TFRecordEncoderDecoder(object):
    """ Define Encoder and Decoder for a specific TFRecord file """
    def __init__(self):
        pass

    def encode_record(self, record_data):
        raise NotImplementedError

    def decode_record(self):
        raise NotImplementedError


class DefaultTFRecordEncoderDecoder(TFRecordEncoderDecoder):
    """ Define Encoder and Decoder for a specific TFRecord file """

    def encode_record(self, record_data, labels_are_numeric=False):
        """ Encode Record to Serialized String """

        # Check Record
        assert all(x in record_data for x in ["id", "labels", "images"]), \
            "Record does not contain all of id/labels/images attributes"

        # Prepare Label fields
        if labels_are_numeric:
            label_dict = wrap_dict_int64_list(record_data['labels'],
                                              prefix='')
        else:
            label_dict = wrap_dict_bytes_list(record_data['labels'],
                                              prefix='')

        # Store Id of Record
        tfrecord_data = {
            'id': wrap_bytes(tf.compat.as_bytes(record_data['id']))
            }

        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=tfrecord_data)

        # Wrap list of images and labels as FeatureLists
        feature_lists = tf.train.FeatureLists(feature_list={
                'images': _bytes_feature_list(record_data['images']),
                **label_dict
                })

        # Wrap again as a TensorFlow Example.
        example = tf.train.SequenceExample(
                context=feature,
                feature_lists=feature_lists)

        # Serialize the data.
        serialized = example.SerializeToString()

        return serialized

    def decode_record(self, serialized_example, output_labels,
                      image_pre_processing_fun=None,
                      image_pre_processing_args=None,
                      n_color_channels=3, choose_random_image=True,
                      decode_images=True,
                      labels_are_numeric=False,
                      one_hot_labels=False,
                      num_classes_list=None
                      ):
        """ Read TFRecord and return dictionary """

        assert (one_hot_labels and labels_are_numeric) or \
               (not one_hot_labels), \
            "One Hot Labels only possible if Labels are numeric"

        assert (not one_hot_labels) or \
               (one_hot_labels and num_classes_list is not None), \
            "One Hot Labels requires num_classes_list"

        # fixed size Features - ID Field
        context_features = {'id': tf.FixedLenFeature([], tf.string)}

        # Decode Labels
        if labels_are_numeric:
            labels = {x:  tf.FixedLenSequenceFeature([], tf.int64)
                      for x in output_labels}
        else:
            labels = {x:  tf.FixedLenSequenceFeature([], tf.string)
                      for x in output_labels}

        # Variable Length (Sequence Features)
        sequence_features = {
            **labels,
            'images': tf.FixedLenSequenceFeature([], tf.string)
            }

        # Parse the serialized data so we get a dict with our data.
        context, sequence = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        # One Hot Encoding
        if one_hot_labels and labels_are_numeric:
            for label, n_cl in zip(output_labels, num_classes_list):
                if n_cl == 2:
                    sequence[label] = tf.reshape(
                        tf.one_hot(sequence[label], n_cl, on_value=0, dtype=tf.int32), [n_cl])
                else:
                    # ignore
                    sequence[label] = tf.reshape(tf.one_hot(sequence[label], n_cl, dtype=tf.int32), [n_cl])
                    # sequence[label] = tf.one_hot(sequence[label], n_cl, dtype=tf.int32)
                    # sequence[label] = tf.squeeze(tf.one_hot(sequence[label], n_cl, dtype=tf.int32))

        if not decode_images:
            return {'images': sequence['images'],
                    'id': context['id'],
                    **{x: sequence[x] for x in output_labels}}

        if choose_random_image:
            # number of images in that record
            n_images = tf.shape(sequence['images'])

            # select a random image of the record
            rand = tf.random_uniform([], minval=0, maxval=n_images[0],
                                     dtype=tf.int32)

            # decode jpeg image to tensor
            image = tf.image.decode_jpeg(sequence['images'][rand],
                                         channels=n_color_channels)

            # Pre-Process image
            if image_pre_processing_fun is not None:
                image_pre_processing_args['image'] = image
                image = image_pre_processing_fun(**image_pre_processing_args)

        else:
            raise NotImplemented("Non-Random Image-Choice not implemented")

        return {'images': image, 'id': context['id'],
                **{x: sequence[x] for x in output_labels}}
