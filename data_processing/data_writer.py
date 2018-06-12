""" Write Data Inventory to Disk """
import random
import os
import math
import logging

import tensorflow as tf

from pre_processing.image_transformations import read_jpeg
from data_processing.utils import slice_generator

logger = logging.getLogger(__name__)


class DatasetWriter(object):
    def __init__(self, tfr_encoder):
        self.tfr_encoder = tfr_encoder
        self.files = dict()

    def encode_to_tfr(
         self, tfrecord_dict,
         output_dir,
         file_prefix,
         image_pre_processing_fun=None,
         image_pre_processing_args=None,
         random_shuffle_before_save=True,
         overwrite_existing_files=True,
         max_records_per_file=None):
        """ Export TFRecord Dict to a TFRecord file """

        self.tfrecord_dict = tfrecord_dict
        self.image_pre_processing_fun = image_pre_processing_fun
        self.image_pre_processing_args = image_pre_processing_args
        self.random_shuffle_before_save = random_shuffle_before_save
        self.file_prefix = file_prefix
        self.files[file_prefix] = list()

        logger.info("Starting to Encode Dict")

        if not isinstance(tfrecord_dict, dict):
            logger.error("tfrecord_dict must be a dictionary")
            raise ValueError("tfrecord_dict must be a dictionary")

        # Sort records to ensure the records are split into files
        # equally each time
        record_ids = list(tfrecord_dict.keys())
        record_ids.sort()
        n_records = len(tfrecord_dict.keys())

        logger.info("Start Writing Records to TFRecord-File - Total %s" %
                    n_records)

        # Generate output file names
        if max_records_per_file is None:
            n_files = 1
        else:
            n_files = math.ceil(n_records / max_records_per_file)

        output_paths = list()
        for i in range(0, n_files):
            file_name = file_prefix + '_%03d.tfrecord' % i
            output_paths.append(os.path.join(*[output_dir, file_name]))

        slices = slice_generator(n_records, n_files)

        # Write each file
        for f_id, output_file in enumerate(output_paths):

            # generate record slices for each file
            start_slice, end_slice = next(slices)
            file_record_ids = record_ids[start_slice:end_slice]

            # check if file already exists
            file_exists = os.path.exists(output_file)

            if file_exists and not overwrite_existing_files:
                logger.info("File: %s exists - not gonna overwrite" %
                            output_file)
                self.files[file_prefix].append(output_file)
            else:
                self._write_to_file(output_file, file_record_ids)

    def _write_to_file(self, output_file, record_ids):
        """ Write a TFR File """

        # Create and Write Records to TFRecord file
        logger.info("Start Writing %s" % output_file)
        n_records = len(record_ids)
        successfull_writes = 0

        # Randomly shuffle records before saving, this is better for
        # model training
        if self.random_shuffle_before_save:
            random.seed(123)
            random.shuffle(record_ids)

        with tf.python_io.TFRecordWriter(output_file) as writer:

            for i, record_id in enumerate(record_ids):

                if i % 1000 == 0:
                    logger.info("Wrote %s / %s records" % (i, n_records))

                record_data = self.tfrecord_dict[record_id]

                # Process all images in a record
                raw_images = list()
                for image_path in record_data['image_paths']:
                    try:
                        if self.image_pre_processing_fun is not None:
                            self.image_pre_processing_args['image'] = image_path
                            image_raw = self.image_pre_processing_fun(
                                 **self.image_pre_processing_args)
                        else:
                            image_raw = read_jpeg(image_path)

                    except Exception as e:
                        logger.debug("Failed to read file: %s , error %s" %
                                     (image_path, str(e)))
                        continue

                    raw_images.append(image_raw)

                # check if at least one image is available
                if len(raw_images) == 0:
                    logger.info("Discarding record %s - no image avail" %
                                record_id)
                    continue

                record_data['images'] = raw_images

                serialized_record = self.tfr_encoder(record_data)

                # Write the serialized data to the TFRecords file.
                writer.write(serialized_record)
                successfull_writes += 1

            logger.info(
                "Finished Writing Records to TFRecord - Wrote %s of %s" %
                (successfull_writes, n_records))

            self.files[self.file_prefix].append(output_file)
