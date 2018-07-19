""" Class To Encode and Decode TFRecords"""
import logging

import tensorflow as tf

from data.utils import (
        wrap_int64, wrap_bytes, wrap_dict_bytes_list,
        _bytes_feature_list,
        _bytes_feature_list_str,
        wrap_dict_int64_list)

logger = logging.getLogger(__name__)


class TFRecordEncoderDecoder(object):
    """ Define Encoder and Decoder for a specific TFRecord file """
    def __init__(self):
        logger.info("Initializing TFRecordEncoderDecoder")

    def encode_record(self, record_data):
        raise NotImplementedError

    def decode_record(self):
        raise NotImplementedError


class DefaultTFRecordEncoderDecoder(TFRecordEncoderDecoder):
    """ Default TFREncoder / Decoder """

    def _convert_to_tfr_data_format(self, record):
        """ Convert a record to a tfr format """

        id = record['id']
        n_images = record['n_images']
        n_labels = record['n_labels']
        image_paths = record['image_paths']
        meta_data = record['meta_data']
        label_text = record['labelstext']
        labels = {k: v for k, v in record.items() if 'label/' in k}

        tfr_data = {
            "id": wrap_bytes(tf.compat.as_bytes(id)),
            "n_images": wrap_int64(n_images),
            "n_labels": wrap_int64(n_labels),
            "image_paths": _bytes_feature_list_str(image_paths),
            "meta_data": wrap_bytes(tf.compat.as_bytes(meta_data)),
            "labelstext": wrap_bytes(tf.compat.as_bytes(label_text)),
            "images": _bytes_feature_list(record['images']),
            **wrap_dict_int64_list(labels)
        }

        return tfr_data

    def encode_record(self, record_data):
        """ Encode Record to Serialized String """

        tfr_data_dict = self._convert_to_tfr_data_format(record_data)

        feature_attributes = set(['id', 'n_images', 'n_labels',
                                  'meta_data', 'labelstext'])

        feature_list_attributes = tfr_data_dict.keys() - feature_attributes

        # Wrap the data as TensorFlow Features
        feature_dict = {k: v for k, v in tfr_data_dict.items()
                        if k in feature_attributes}
        feature = tf.train.Features(feature=feature_dict)

        # Wrap lists as FeatureLists
        feature_list_dict = {k: v for k, v in tfr_data_dict.items()
                             if k in feature_list_attributes}
        feature_lists = tf.train.FeatureLists(feature_list=feature_list_dict)

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
                      n_color_channels=3,
                      choose_random_image=True,
                      decode_images=True,
                      return_only_ml_data=True,
                      only_return_one_label=True
                      ):
        """ Read TFRecord and return dictionary
        """

        # fixed size Features - ID and labels
        if return_only_ml_data:
            context_features = {
                'id': tf.FixedLenFeature([], tf.string)
                }
        else:
            context_features = {
                'id': tf.FixedLenFeature([], tf.string),
                'n_images': tf.FixedLenFeature([], tf.int64),
                'n_labels': tf.FixedLenFeature([], tf.int64),
                'meta_data': tf.FixedLenFeature([], tf.string),
                'labelstext': tf.FixedLenFeature([], tf.string)
                }

        # Extract labels
        label_names = ['label/' + l for l in output_labels]
        label_features = {k: tf.FixedLenSequenceFeature([], tf.int64)
                          for k in label_names}

        if return_only_ml_data:
            sequence_features = {
                'images': tf.FixedLenSequenceFeature([], tf.string),
                **label_features
                }
        else:
            sequence_features = {
                'images': tf.FixedLenSequenceFeature([], tf.string),
                'image_paths': tf.FixedLenSequenceFeature([], tf.string),
                **label_features
                }

        # Parse the serialized data so we get a dict with our data.
        context, sequence = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        if only_return_one_label:
            parsed_labels = {k: v[0] for k, v in sequence_features.items()
                             if k in 'label/'}
        else:
            parsed_labels = {k: v for k, v in sequence_features.items()
                             if k in 'label/'}

        if not decode_images:
            return {**{y: context[y] for y in context_features.keys()},
                    **{x: sequence[x] for x in sequence_features.keys()
                       if x not in 'label/'},
                    **parsed_labels}

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

        return ({'images': image},
                {**{y: context[y] for y in context_features.keys()},
                **{x: sequence[x] for x in sequence_features.keys()
                   if x not in ['label/', 'images']},
                **parsed_labels})


class SingleObsTFRecordEncoderDecoder(TFRecordEncoderDecoder):
    """ Define Encoder and Decoder for a specific TFRecord file """

    def encode_record(self, record_data):
        """ Encode Record to Serialized String """

        # Check Record
        assert all(x in record_data for x in ["id", "images"]), \
            "Record does not contain all of id/images attributes"

        label_entries = record_data.keys() - set(['id', 'images'])

        # Store Id and labels of Record
        tfrecord_data = {
            'id': wrap_bytes(tf.compat.as_bytes(record_data['id'])),
            **{l: wrap_int64(record_data[l])
                for l in label_entries}
            }

        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=tfrecord_data)

        # Wrap list of images and labels as FeatureLists
        feature_lists = tf.train.FeatureLists(feature_list={
                'images': _bytes_feature_list(record_data['images'])
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
                      one_hot_labels=False,
                      num_classes_list=None
                      ):
        """ Read TFRecord and return dictionary """

        # fixed size Features - ID and labels
        context_features = {
            'id': tf.FixedLenFeature([], tf.string),
            **{l: tf.FixedLenFeature([], tf.int64) for l in output_labels}
            }

        # Variable Length (images)
        sequence_features = {
            'images': tf.FixedLenSequenceFeature([], tf.string)
            }

        # Parse the serialized data so we get a dict with our data.
        context, sequence = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        # One Hot Encoding
        if one_hot_labels:
            for label, n_cl in zip(output_labels, num_classes_list):
                if n_cl == 2:
                    context[label] = tf.reshape(
                        tf.one_hot(sequence[label], n_cl,
                                   on_value=0, dtype=tf.int32), [n_cl])
                else:
                    # ignore
                    context[label] = tf.reshape(
                        tf.one_hot(context[label],
                                   n_cl, dtype=tf.int32), [n_cl])

        if not decode_images:
            return {'images': sequence['images'],
                    'id': context['id'],
                    **{x: context[x] for x in output_labels}}

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
                **{x: context[x] for x in output_labels}}


class MultiLabelTFRecordEncoderDecoder(TFRecordEncoderDecoder):
    """ Define Encoder and Decoder for a specific TFRecord file """

    def encode_record(self, record_data, labels_are_numeric=False):
        """ Encode Record to Serialized String """

        # Check Record
        assert all(x in record_data for x in ["id", "images", "n_labels"]), \
            "Record does not contain all of id/n_labels/images attributes"

        # label fields
        label_fields = record_data.keys() - set(["id", "images", "n_labels"])
        label_dict = {k: record_data[k] for k in label_fields}

        # Prepare Label fields
        if labels_are_numeric:
            label_dict = wrap_dict_int64_list(label_dict,
                                              prefix='')
        else:
            label_dict = wrap_dict_bytes_list(label_dict,
                                              prefix='')

        # Store Id and n_labels of Record
        tfrecord_data = {
            'id': wrap_bytes(tf.compat.as_bytes(record_data['id'])),
            'n_labels': wrap_int64(tf.compat.as_bytes(record_data['n_labels']))
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
        context_features = {'id': tf.FixedLenFeature([], tf.string),
                            'n_labels': tf.FixedLenFeature([], tf.int64)}

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


class PatnheraTFRecordEncoderDecoder(TFRecordEncoderDecoder):
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
