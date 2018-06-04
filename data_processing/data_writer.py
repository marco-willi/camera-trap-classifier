""" Write Data Inventory to Disk """
import random
import os
import logging

import tensorflow as tf

from pre_processing.image_transformations import read_jpeg

logger = logging.getLogger(__name__)


class DatasetWriter(object):
    def __init__(self, tfr_encoder):
        self.tfr_encoder = tfr_encoder

    def encode_to_tfr(
         self, tfrecord_dict, output_file,
         image_pre_processing_fun=None,
         image_pre_processing_args=None,
         random_shuffle_before_save=True,
         overwrite_existing_file=True,
         prefix_to_labels=''):
        """ Export TFRecord Dict to a TFRecord file """

        if os.path.exists(output_file) and not overwrite_existing_file:
            logger.info("File: %s exists - not gonna overwrite" % output_file)
            return None

        logger.info("Starting to Encode Dict")

        if not isinstance(tfrecord_dict, dict):
            logger.error("tfrecord_dict must be a dictionary")
            raise ValueError("tfrecord_dict must be a dictionary")

        # Create and Write Records to TFRecord file
        with tf.python_io.TFRecordWriter(output_file) as writer:

            record_ids = list(tfrecord_dict.keys())

            # Randomly shuffle records before saving, this is better for
            # model training
            if random_shuffle_before_save:
                random.seed(123)
                random.shuffle(record_ids)

            n_records = len(tfrecord_dict.keys())
            logger.info("Start Writing Records to TFRecord-File - Total %s" %
                        n_records)

            # Loop over all records and write to TFRecord
            successfull_writes = 0
            for i, record_id in enumerate(record_ids):

                if i % 1000 == 0:
                    logger.info("Wrote %s / %s files" % (i, n_records))

                record_data = tfrecord_dict[record_id]

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
