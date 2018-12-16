""" Class To Encode and Decode TFRecords"""
import logging

import tensorflow as tf

from camera_trap_classifier.data.utils import (
        wrap_int64, wrap_bytes, wrap_dict_bytes_list, wrap_dict_int64_list,
        _bytes_feature_list,
        _bytes_feature_list_str)
from camera_trap_classifier.data.image import decode_image_bytes_1D

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
        labels_num = {k: v for k, v in record.items() if 'label_num/' in k}

        label_features = wrap_dict_bytes_list(labels)
        label_num_features = wrap_dict_int64_list(labels_num)

        tfr_data = {
            "id": wrap_bytes(tf.compat.as_bytes(id)),
            "n_images": wrap_int64(n_images),
            "n_labels": wrap_int64(n_labels),
            "image_paths": _bytes_feature_list_str(image_paths),
            "meta_data": wrap_bytes(tf.compat.as_bytes(meta_data)),
            "labelstext": wrap_bytes(tf.compat.as_bytes(label_text)),
            "images": _bytes_feature_list(record['images']),
            **label_features,
            **label_num_features
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

    def decode_record(self, serialized_example,
                      output_labels,
                      label_lookup_dict=None,
                      image_pre_processing_fun=None,
                      image_pre_processing_args=None,
                      image_choice_for_sets='random',
                      decode_images=True,
                      numeric_labels=False,
                      return_only_ml_data=True,
                      only_return_one_label=True
                      ):
        """ Decode TFRecord and return dictionary """
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

        # Extract labels (string and numeric)
        label_names = ['label/' + l for l in output_labels]
        label_features = {k: tf.FixedLenSequenceFeature([], tf.string)
                          for k in label_names}

        label_num_names = ['label_num/' + l for l in output_labels]
        label_num_features = {k: tf.FixedLenSequenceFeature([], tf.int64)
                              for k in label_num_names}

        if return_only_ml_data:
            if numeric_labels:
                sequence_features = {
                    'images': tf.FixedLenSequenceFeature([], tf.string),
                    **label_num_features
                    }
            else:
                sequence_features = {
                    'images': tf.FixedLenSequenceFeature([], tf.string),
                    **label_features
                    }
        else:
            sequence_features = {
                'images': tf.FixedLenSequenceFeature([], tf.string),
                'image_paths': tf.FixedLenSequenceFeature([], tf.string),
                **label_features,
                **label_num_features
                }

        # Parse the serialized data so we get a dict with our data.
        context, sequence = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        # determine label prefix for either numeric or string labels
        if numeric_labels:
            label_prefix = 'label_num/'
        else:
            label_prefix = 'label/'

        # Wheter to return only the labels of the first observation or all
        # and wheter to map string labels to integers using a lookup table
        if only_return_one_label:
            if label_lookup_dict is not None and not numeric_labels:
                parsed_labels = {
                    k: tf.reshape(label_lookup_dict[k].lookup(v[0]), [1])
                    for k, v in sequence.items() if label_prefix in k}
            else:
                parsed_labels = {
                    k: v[0]
                    for k, v in sequence.items() if label_prefix in k}
        else:
            if label_lookup_dict is not None and not numeric_labels:
                parsed_labels = {
                    k: label_lookup_dict[k].lookup(v)
                    for k, v in sequence.items() if label_prefix in k}
            else:
                parsed_labels = {
                    k: v
                    for k, v in sequence.items() if label_prefix in k}

        if not decode_images:
            return {**{k: v for k, v in context.items()},
                    **{k: v for k, v in sequence.items()
                       if label_prefix not in k},
                    **parsed_labels}

        # decode 1-D tensor of raw images
        image = decode_image_bytes_1D(
                    sequence['images'],
                    **image_pre_processing_args)

        # Pre-Process image
        if image_pre_processing_fun is not None:
            image_pre_processing_args['image'] = image
            image = image_pre_processing_fun(**image_pre_processing_args)

        return ({'images': image},
                {**{k: v for k, v in context.items()},
                 **{k: v for k, v in sequence.items()
                 if label_prefix not in k and 'images' not in k},
                **parsed_labels})
