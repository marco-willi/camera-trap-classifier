""" Write Data Inventory to Disk as TFRecord files """
import random
import os
import math
import time
import logging
from multiprocessing import Process, Manager

import tensorflow as tf

from data.image import read_jpeg
from data.utils import slice_generator, estimate_remaining_time

logger = logging.getLogger(__name__)


class DatasetWriter(object):
    def __init__(self, tfr_encoder):
        self.tfr_encoder = tfr_encoder
        self.files = dict()

    def encode_to_tfr(
         self, tfrecord_dict,
         output_dir,
         file_prefix,
         image_root_path='',
         image_pre_processing_fun=None,
         image_pre_processing_args=None,
         random_shuffle_before_save=True,
         overwrite_existing_files=True,
         max_records_per_file=None,
         write_tfr_in_parallel=False,
         process_images_in_parallel=False,
         process_images_in_parallel_size=100,
         processes_images_in_parallel_n_processes=4):
        """ Export TFRecord Dict to a TFRecord file """

        self.tfrecord_dict = tfrecord_dict
        self.image_pre_processing_fun = image_pre_processing_fun
        self.image_pre_processing_args = image_pre_processing_args
        self.random_shuffle_before_save = random_shuffle_before_save
        self.file_prefix = file_prefix
        self.image_root_path = image_root_path
        self.files[file_prefix] = list()
        self.write_tfr_in_parallel = write_tfr_in_parallel
        self.process_images_in_parallel = process_images_in_parallel
        self.process_images_in_parallel_size = process_images_in_parallel_size
        self.processes_images_in_parallel_n_processes = \
            processes_images_in_parallel_n_processes

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
            file_name = '%s_%03d-of-%03d.tfrecord' % (file_prefix, i+1, n_files)
            output_paths.append(os.path.join(*[output_dir, file_name]))

        # processes list if parallel processing is enabled
        if self.write_tfr_in_parallel:
            processes_list = list()

        slices = slice_generator(n_records, n_files)

        # Write each file
        for f_id, (start_i, end_i) in enumerate(slices):
            output_file = output_paths[f_id]
            # generate record slices for each file
            file_record_ids = record_ids[start_i:end_i]

            # check if file already exists
            file_exists = os.path.exists(output_file)

            if file_exists and not overwrite_existing_files:
                logger.info("File: %s exists - not gonna overwrite" %
                            output_file)
                self.files[file_prefix].append(output_file)
            else:
                if self.write_tfr_in_parallel:
                    pr = Process(target=self._write_to_file,
                                 args=(output_file, file_record_ids))
                    pr.start()
                    processes_list.append(pr)
                else:
                    if self.process_images_in_parallel:
                        self._write_to_file_parallel(output_file,
                                                     file_record_ids)
                    else:
                        self._write_to_file(output_file, file_record_ids)
            self.files[self.file_prefix].append(output_file)
        # start all processes
        if self.write_tfr_in_parallel:
            for p in processes_list:
                p.join()

    def _serialize_record_batch(self, record_batch, outlist):
        """ Serialize a list of records """
        for record_data in record_batch:
            serialized_record = self._serialize_record(record_data)
            if serialized_record is not None:
                outlist.append(serialized_record)

    def _serialize_record(self, record_data):
        """ Serialize a single record """
        # Process all images in a record
        raw_images = list()
        for image_path in record_data['image_paths']:
            image_path_full = os.path.join(self.image_root_path,
                                           image_path)
            try:
                if self.image_pre_processing_fun is not None:
                    self.image_pre_processing_args['image'] = \
                        image_path_full
                    image_raw = self.image_pre_processing_fun(
                         **self.image_pre_processing_args)
                else:
                    image_raw = read_jpeg(image_path_full)

            except Exception as e:
                logger.debug("Failed to read image: %s , error %s" %
                             (image_path_full, str(e)))
                continue

            raw_images.append(image_raw)

        # check if at least one image is available
        if len(raw_images) == 0:
            return None

        record_data['images'] = raw_images

        serialized_record = self.tfr_encoder(record_data)

        return serialized_record

    def _write_to_file(self, output_file, record_ids):
        """ Write a TFR File """

        # Create and Write Records to TFRecord file
        logger.info("Start Writing %s" % output_file)
        n_records = len(record_ids)
        successfull_writes = 0
        start_time = time.time()

        # Randomly shuffle records before saving, this is better for
        # model training
        if self.random_shuffle_before_save:
            random.seed(123)
            random.shuffle(record_ids)

        with tf.python_io.TFRecordWriter(output_file) as writer:

            for i, record_id in enumerate(record_ids):

                if i % 1000 == 0:
                    est_t = estimate_remaining_time(start_time, n_records, i)
                    logger.info(
                        "Wrote %s / %s records (estimated time remaining: %s)"
                        % (i, n_records, est_t))

                record_data = self.tfrecord_dict[record_id]

                serialized_record = self._serialize_record(record_data)

                if serialized_record is None:
                    logger.info("Discarding record %s - no image avail" %
                                record_id)
                    continue

                # Write the serialized data to the TFRecords file.
                writer.write(serialized_record)
                successfull_writes += 1

            logger.info(
                "Finished Writing Records to %s - Wrote %s/%s" %
                (output_file, successfull_writes, n_records))

    def _write_to_file_parallel(self, output_file, record_ids):
        """ Write a TFR File with parallel image processing """

        # Create and Write Records to TFRecord file
        logger.info("Start Writing %s" % output_file)
        n_records = len(record_ids)
        successfull_writes = 0
        start_time = time.time()

        # Randomly shuffle records before saving, this is better for
        # model training
        if self.random_shuffle_before_save:
            random.seed(123)
            random.shuffle(record_ids)

        # divide writes into batches
        max_batch_size = min(self.process_images_in_parallel_size, n_records)
        n_batches = (n_records // max_batch_size) \
            + min(n_records % max_batch_size, 1)
        slices = slice_generator(n_records, n_batches)

        with tf.python_io.TFRecordWriter(output_file) as writer:

            for batch_i, (start_i, end_i) in enumerate(slices):

                # Divide current batch to multiple processes
                record_batch = [self.tfrecord_dict[x] for x in
                                record_ids[start_i: end_i]]
                n_records_in_batch = len(record_batch)

                process_slices = slice_generator(
                    n_records_in_batch,
                    self.processes_images_in_parallel_n_processes)

                manager = Manager()
                serialized_records = manager.list()

                processes_list = list()

                for p_i, (batch_start_i, batch_end_i) in enumerate(process_slices):

                    process_batch = record_batch[batch_start_i: batch_end_i]
                    pr = Process(target=self._serialize_record_batch,
                                 args=(process_batch, serialized_records))
                    pr.start()
                    processes_list.append(pr)

                for p in processes_list:
                    p.join()

                # Write the serialized data to the TFRecords file.
                for serialized_record in serialized_records:
                    writer.write(serialized_record)
                    successfull_writes += 1
                est_t = estimate_remaining_time(start_time, n_records,
                                                successfull_writes)
                logger.info("Wrote %s / %s records - estimated time remaining: %s - file: %s)" %
                            (successfull_writes, n_records, est_t, output_file))

            logger.info(
                "Finished Writing Records to %s - Wrote %s/%s" %
                (output_file, successfull_writes, n_records))
