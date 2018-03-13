""" Class To Encode and Decode TFRecords"""
from config.config import logging
import tensorflow as tf
from data_processing.utils import (
        wrap_int64, wrap_bytes, wrap_dict_bytes_list,
        wrap_dict_bytes_str,
        _bytes_feature_list,
        wrap_dict_int64_list)
import random
from pre_processing.image_transformations import read_jpeg
from collections import OrderedDict


class TFRecordEncoderDecoder(object):
    """ Define Encoder and Decoder for a specific TFRecord file """
    def __init__(self):
        pass

    def encode_dict_to_tfr(self, data_inventory, output_file):
        raise NotImplementedError

    def get_tfr_decoder(self):
        raise NotImplementedError


class CamCatTFRecordEncoderDecoder(TFRecordEncoderDecoder):
    """ Camera Catalogue Encoder Decoder Version """

    def get_tfr_decoder(self):
        """ Get Decoder """
        return self._decode_tfrecord

    def read_write_tfr(input_file, output_file):
        """ Read and Write TFRecod File """
        raise NotImplementedError

    def encode_dict_to_tfr(self, data_inventory, output_file,
                           image_pre_processing_fun=None,
                           image_pre_processing_args=None,
                           random_shuffle_before_save=True):
        """ Export Dictionary of Record to a TFRecord file """

        logging.info("Starting to Encode Inventory to Dictionary")

        try:
            all_label_types = data_inventory.get_all_label_types()
        except:
            raise ValueError("data_inventory must be of class DataInventory")

        logging.info("Found following label types: %s" % all_label_types)

        n_records = data_inventory.get_number_of_records()
        logging.info("Found %s records in inventory" % n_records)

        # Create and Write Records to TFRecord file
        with tf.python_io.TFRecordWriter(output_file) as writer:

            record_ids = data_inventory.get_all_record_ids()

            # Randomly shuffle records before saving, this is better for
            # model training
            if random_shuffle_before_save:
                random.seed(123)
                random.shuffle(record_ids)

            logging.info("Start Writing Record to TFRecord - Total %s" %
                         n_records)

            # Loop over all records and write to TFRecord
            successfull_writes = 0
            for i, (record_id) in enumerate(record_ids):

                if i % 1000 == 0:
                    logging.info("Wrote %s / %s files" % (i, n_records))

                record_data = data_inventory.get_record_id_data(record_id)

                # Process all images in a record
                raw_images = list()
                for image_path in record_data['images']:
                    try:
                        if image_pre_processing_fun is not None:
                            image_pre_processing_args['image'] = image_path
                            image_raw = image_pre_processing_fun(
                                 **image_pre_processing_args)
                        else:
                            image_raw = read_jpeg(image_path)

                    except Exception as e:
                        logging.debug("Failed to read file: %s , error %s" %
                                      (image_path, str(e)))
                        continue

                    raw_images.append(image_raw)

                # check if at least one image is available
                if len(raw_images) == 0:
                    logging.info("Discarding record %s - no image avail" %
                                 record_id)
                    data_inventory.remove_record(record_id)
                    continue

                # Prepare Meta-Data Fields
                if 'meta_data' in record_data:

                    meta_data = wrap_dict_bytes_str(record_data['meta_data'],
                                                    prefix='meta_data/')
                else:
                    meta_data = {'meta_data':
                                 wrap_bytes(tf.compat.as_bytes(''))}

                # Prepare Label fields
                label_dict = wrap_dict_bytes_list(record_data['labels'],
                                                  prefix='labels/')

                # store all information of a record
                tfrecord_data = {
                    'id': wrap_bytes(tf.compat.as_bytes(record_id)),
                    **meta_data,
                    'n_images': wrap_int64(len(raw_images))
                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=tfrecord_data)

                # Wrap list of images as FeatureList
                feature_lists = tf.train.FeatureLists(feature_list={
                        'images': _bytes_feature_list(raw_images),
                        **label_dict
                        })

                # Wrap again as a TensorFlow Example.
                example = tf.train.SequenceExample(
                        context=feature,
                        feature_lists=feature_lists)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
                successfull_writes += 1

            logging.info(
                "Finished Writing Records to TFRecord - Wrote %s of %s" %
                (successfull_writes, n_records))

    def serialize_split_tfr_record(self, record_dict):
        """ Serialize Single Record from Dict """

        # Write new Record
        label_dict = wrap_dict_int64_list(record_dict['labels'],
                                          prefix='')

        # store all information of a record
        tfrecord_data = {
            'id': wrap_bytes(tf.compat.as_bytes(record_dict['id']))
            }

        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=tfrecord_data)

        # Wrap list of images as FeatureList
        feature_lists = tf.train.FeatureLists(feature_list={
                'images':  _bytes_feature_list(record_dict['images']),
                **label_dict
                })

        # Wrap again as a TensorFlow Example.
        example = tf.train.SequenceExample(
                context=feature,
                feature_lists=feature_lists)

        # Serialize the data.
        serialized = example.SerializeToString()

        return serialized

    def _decode_ids_and_labels(self, tfr_file, label_types):
        # get all record ids and labels
        data_iterator = tf.python_io.tf_record_iterator(tfr_file)
        records_info_tf = OrderedDict()

        labels = ['labels/' + label for label in label_types]

        # Iterate over TFrecord file
        for data_record in data_iterator:
            record_id, label_types = tf.parse_single_sequence_example(
                serialized=data_record,
                context_features={
                    'id': tf.FixedLenFeature([], tf.string)},
                sequence_features={
                    **{x: tf.FixedLenSequenceFeature([], tf.string)
                        for x in labels}})

            # store labels for each record
            records_info_tf[record_id['id']] = label_types

        # actually collect the data
        with tf.Session() as sess:
            records_info = sess.run(records_info_tf)

        return records_info

    def _decode_labels_and_images(self, serialized_example,
                                  output_labels, **kwargs):
        """ Read and Decode Labels and Images """

        # fixed size data
        context_features = {
                'id': tf.FixedLenFeature([], tf.string)
                }

        # variable sized features - number of images
        sequence_features = {
            **{x:  tf.FixedLenSequenceFeature([], tf.string)
                for x in output_labels},
            'images': tf.FixedLenSequenceFeature([], tf.string)
            }

        # Parse the serialized data so we get a dict with our data.
        context, sequence = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        return {'id': context['id']}, \
               {'images': sequence['images'],
                **{x: sequence[x] for x in output_labels}}

    def _decode_tfrecord(self, serialized_example, output_labels,
                         image_pre_processing_fun,
                         image_pre_processing_args,
                         n_color_channels=3, choose_random_image=True,
                         decode_images=True,
                         numeric_labels=False
                         ):
        """ Read TFRecord and return dictionary """

        # fixed size data
        context_features = {
                'id': tf.FixedLenFeature([], tf.string)
                # 'meta_data': tf.FixedLenFeature([], tf.string),
                # 'n_images': tf.FixedLenFeature([], tf.int64)
                }

        # variable sized features - number of images
        if numeric_labels:
            labels = {x:  tf.FixedLenSequenceFeature([], tf.int64)
                      for x in output_labels}
        else:
            labels = {x:  tf.FixedLenSequenceFeature([], tf.string)
                      for x in output_labels}
        sequence_features = {
            **labels,
            'images': tf.FixedLenSequenceFeature([], tf.string)
            }

        # Parse the serialized data so we get a dict with our data.
        context, sequence = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        if not decode_images:
            return {'images': sequence['images'],
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

        return ({'images': image}, {x: sequence[x] for x in output_labels})

        # # map labels with HashTable
        # def hash_mapper(output_label):
        #
        #     target_label_type = output_label.split('/')[1]
        #     if target_label_type not in class_mapper:
        #         logging.error("Label_Type %s not found in mapper"
        #                       % target_label_type)
        #     keys = class_mapper[target_label_type]['keys']
        #     values = class_mapper[target_label_type]['values']
        #     keys_tf = tf.cast(keys, tf.string)
        #     values_tf = tf.cast(values, tf.int32)
        #     # table = tf.contrib.lookup.HashTable(
        #     #     tf.contrib.lookup.KeyValueTensorInitializer(
        #     #         keys_tf, values_tf), -1)
        #     return keys_tf, values_tf

        # map labels with HashTable
        # def hash_mapper(output_label):
        #     with tf.variable_scope("output_label/" + output_label, reuse=False):
        #         target_label_type = output_label.split('/')[1]
        #         if target_label_type not in class_mapper:
        #             logging.error("Label_Type %s not found in mapper"
        #                           % target_label_type)
        #         keys = class_mapper[target_label_type]['keys']
        #         values = class_mapper[target_label_type]['values']
        #         keys_tf = tf.cast(keys, tf.string)
        #         values_tf = tf.cast(values, tf.int32)
        #         table = tf.contrib.lookup.HashTable(
        #             tf.contrib.lookup.KeyValueTensorInitializer(
        #                 keys_tf, values_tf), -1)
        #         return table


        # labels_mapped = dict()
        # for x in output_labels:
        #     with tf.variable_scope("output_label/" + x, reuse=False):
        #         k, v = hash_mapper(x)
        #
        #         table = tf.contrib.lookup.HashTable(
        #             tf.contrib.lookup.KeyValueTensorInitializer(
        #                 k, v), -1)
        #         labels_mapped[x] = table.lookup(sequence[x])

        #return ({'images': image}, {x: class_mappers[x].lookup(sequence[x]) for x in output_labels})

        #return ({'images': image}, labels_mapped)
