""" Write Data Inventory to Disk """
from config.config import logging
import tensorflow as tf
import random
from pre_processing.image_transformations import read_jpeg
from data_processing.data_inventory import DatasetInventory


class DatasetWriter(object):
    def __init__(self, tfr_encoder):
        self.tfr_encoder = tfr_encoder

    def encode_inventory_to_tfr(
         self, data_inventory, output_file,
         image_pre_processing_fun=None,
         image_pre_processing_args=None,
         random_shuffle_before_save=True):
        """ Export Data Inventory to a TFRecord file """

        logging.info("Starting to Encode Inventory to Dictionary")

        if not isinstance(data_inventory, DatasetInventory):
            raise ValueError("data_inventory must be a DatasetInventory")

        all_label_types = data_inventory.get_all_label_types()
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
            for i, record_id in enumerate(record_ids):

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

                # Create Record to Serialize
                record_to_serialize = dict()
                record_to_serialize['id'] = record_id
                record_to_serialize['labels'] = record_data['labels']
                record_to_serialize['images'] = raw_images

                serialized_record = self.tfr_encoder(record_to_serialize,
                                                     labels_are_numeric=False)

                # Write the serialized data to the TFRecords file.
                writer.write(serialized_record)
                successfull_writes += 1

            logging.info(
                "Finished Writing Records to TFRecord - Wrote %s of %s" %
                (successfull_writes, n_records))
