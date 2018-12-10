""" Class To Encode and Decode TFRecords"""
import logging

import tensorflow as tf

from camera_trap_classifier.data.utils import (
        wrap_int64, wrap_bytes, wrap_dict_bytes_list, wrap_dict_int64_list,
        _bytes_feature_list,
        _bytes_feature_list_str)
from camera_trap_classifier.data.image import (
    gaussian_kernel_2D
)

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
                      image_choice_if_multiple='random',
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
            #return sequence['images']

        if image_choice_if_multiple == 'random':
            image = self._choose_random_image(sequence['images'])
        elif image_choice_if_multiple == 'grayscale_blurring':
            image = self._grayscale_blurring(sequence['images'])
        else:
            raise NotImplemented("Non-Random Image-Choice not implemented")

        # Pre-Process image
        if image_pre_processing_fun is not None:
            image_pre_processing_args['image'] = image
            image = image_pre_processing_fun(**image_pre_processing_args)

        return ({'images': image},
                {**{k: v for k, v in context.items()},
                 **{k: v for k, v in sequence.items()
                 if label_prefix not in k and 'images' not in k},
                **parsed_labels})

    def _choose_random_image(self, image_bytes):
        """ Choose a random image """
        n_images = tf.shape(image_bytes)

        # select a random image of the record
        rand = tf.random_uniform([], minval=0, maxval=n_images[0],
                                 dtype=tf.int32)

        # decode image to tensor
        image = tf.image.decode_jpeg(image_bytes[rand])

        return image

    def _decode_image_bytes_example(self, image_bytes, n_colors=3):
        """ Input is one TFRecord Exaample
            Example with three images:
                TensorShape([Dimension(1), Dimension(3)])
            Example Output:
                TensorShape([Dimension(3), Dimension(375),
                             Dimension(500), Dimension(3)])
        """
        images = tf.map_fn(
                    lambda x: tf.image.decode_jpeg(x, channels=n_colors),
                    image_bytes, dtype=tf.uint8)
        return images

    def _stack_images_to_3D(self, image_tensor):
        """ Stack images """
        input_shape = tf.shape(image_tensor)
        if input_shape[-1] == 1:
            target_shape = image_tensor.get_shape().as_list()
            target_shape[-1] = 3
            image_tensor = tf.broadcast_to(image_tensor, target_shape)
        elif input_shape[-1] == 2:
            image_tensor = tf.stack([
                image_tensor[:, :, 0],
                image_tensor[:, :, 1],
                image_tensor[:, :, 1]], 2)
        return image_tensor

    def _blurr_imgs(self, img_batch):
        """ Blurr image batch with Gaussian Filter """
        with tf.variable_scope("gauss_kernel"):
            gauss_kernel = gaussian_kernel_2D(sigma=2)
            gauss_kernel = tf.expand_dims(tf.expand_dims(gauss_kernel, -1), -1)

        img_batch = tf.cast(img_batch, tf.float32)
        img_batch_blurred = tf.nn.conv2d(
            img_batch,
            filter=gauss_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_cudnn_on_gpu=True,
            data_format='NHWC'
        )

        return img_batch_blurred

    def _grayscale_blurring(self, image_bytes):
        """ Get and convert all images to grayscale """

        # Grayscale image batch tensor (4-D, NHWC)
        imgs = self._decode_image_bytes_example(image_bytes, n_colors=1)

        # Apply Gaussian Blurring
        # Batch of 1-N blurred images
        imgs_blurred = self._blurr_imgs(imgs)

        # Stack into RGB image, handle cases when there is only 1 or 2 images
        image = tf.transpose(tf.squeeze(imgs_blurred, -1), perm=[1, 2, 0])
        image = self._stack_images_to_3D(image)

        image = tf.cast(image, tf.uint8)

        return image
