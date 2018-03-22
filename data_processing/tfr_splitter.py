""" Class To Split TFRecords """
from random import shuffle
import os
from collections import OrderedDict

import tensorflow as tf

from config.config import logging
from data_processing.utils import id_to_zero_one, n_records_in_tfr
from data_processing.data_reader import DatasetReader
from data_processing.data_inventory import DatasetInventory
from data_processing.label_handler import LabelHandler


class TFRecordSplitter(object):
    """ Splits TFRecord files """

    def __init__(self, files_to_split, tfr_encoder, tfr_decoder):
        self.files_to_split = files_to_split
        self.tfr_encoder = tfr_encoder
        self.tfr_decoder = tfr_decoder
        self.split_files = None
        self.split_props = None
        self.split_names = None
        self.class_mapping = None

    def get_split_paths(self):
        """ Get Split-Paths and Split-Names """
        return {k: v for k, v in zip(self.split_names, self.split_files)}

    def log_record_numbers_per_file(self):
        """ print record numbers per file """
        for f, n in zip(self.split_files, self.split_names):
            logging.info("File: %s has %s records" % (n, n_records_in_tfr(f)))

    def get_record_numbers_per_file(self):
        """ get record numbers per file in dict """
        return {n: n_records_in_tfr(f) for f, n in
                zip(self.split_files, self.split_names)}

    def _check_and_clean_input(self):
        """ Check and Clean Input """

        # Check Split Proportions
        assert sum(self.split_props) == 1, \
            "sum of split_prop has to be 1, is: %s" % sum(self.split_props)

        # Check Split Types
        assert type(self.split_names) is list and \
            type(self.split_props) is list, \
            "splits and split_prop must be lists"

        # Check Split Names
        assert len(self.split_names) == len(self.split_props), \
            "Number of splits must be equal to number of split_prop"

        # Rename class label types
        if self.class_mapping is not None:
            for label_type in self.class_mapping.keys():
                self.class_mapping['labels/' + label_type] = \
                    self.class_mapping.pop(label_type)

        # Clean output labels
        self.output_labels_clean = ['labels/' + x for x in self.output_labels]

        # check keep / remove labels
        if self.keep_only_labels is not None:
            keep_only_labels_clean = dict()
            for label_type, labels in self.keep_only_labels.items():
                labels_list = self._convert_to_list(self.keep_only_labels)
                keep_only_labels_clean['labels/' + label_type] = labels_list

            self.keep_only_labels = keep_only_labels_clean

        if self.remove_label_types is not None:
            self.remove_label_types = \
             ['labels/' + x for x in self.remove_label_types]

    def split_tfr_file(self, output_path_main, output_prefix,
                       split_names, split_props, output_labels,
                       class_mapping=None,
                       balanced_sampling_min=False,
                       balanced_sampling_label_type=None,
                       overwrite_existing_files=True,
                       keep_only_labels=None,
                       remove_label_types=None):
        """ Split a TFR file according to split proportions """

        self.split_names = split_names
        self.split_props = split_props
        self.class_mapping = class_mapping
        self.output_labels = output_labels
        self.keep_only_labels = keep_only_labels
        self.remove_label_types = remove_label_types

        self._check_and_clean_input()

        # Create Output File Names
        output_file_names = [output_path_main + output_prefix + '_' +
                             s + '.tfrecord' for s in self.split_names]

        self.split_files = output_file_names

        # get all ids and their labels from the input file
        dataset_reader = DatasetReader(self.tfr_decoder)

        iterator = dataset_reader.get_iterator(
             self.files_to_split, batch_size=128, is_train=False, n_repeats=1,
             output_labels=output_labels,
             buffer_size=2048,
             decode_images=False,
             labels_are_numeric=False,
             max_multi_label_number=None)

        id_label_dict = OrderedDict()
        with tf.Session() as sess:
            while True:
                try:
                    batch_data = sess.run(iterator)
                    self._extract_id_labels(id_label_dict,
                                            batch_data,
                                            self.output_labels_clean)
                except tf.errors.OutOfRangeError:
                    break

        # convert label dict to inventory
        data_inv = self._convert_id_label_dict_to_inventory(id_label_dict)

        # keep only specific labels
        if self.keep_only_labels is not None:
            data_inv.label_handler.keep_only_labels(self.keep_only_labels)

        # keep only specific label types
        if self.remove_label_types is not None:
            data_inv.label_handler.remove_label_types(self.remove_label_types)

        # map labels if specified
        if self.class_mapping is not None:
            data_inv.label_handler.map_labels(self.class_mapping)

        # change labels to numeric
        data_inv.label_handler.map_labels_to_numeric()
        self.label_to_numeric_mapper = data_inv.label_handler.labels_to_numeric

        # Check if all files exist
        if not overwrite_existing_files:
            if all([os.path.exists(x) for x in output_file_names]):
                logging.info("Files: %s exist - not gonna overwrite"
                             % output_file_names)
                return None

        # Convert data inventory back to id_label_dict
        id_label_dict = self._convert_inventory_to_id_label_dict(data_inv)

        # assign each id to a splitting value
        id_to_split_assignments = self._assign_id_to_set(
            id_label_dict, split_names,
            split_props, balanced_sampling_min,
            balanced_sampling_label_type)

        # Write TFrecord files for each split
        for i, split in enumerate(self.split_names):

            iterator = dataset_reader.get_iterator(
                 self.files_to_split, batch_size=128,
                 is_train=False, n_repeats=1,
                 output_labels=output_labels,
                 buffer_size=2048,
                 decode_images=False,
                 labels_are_numeric=False,
                 max_multi_label_number=None)

            # Write Split File
            logging.info("Start writing file %s" % output_file_names[i])
            with tf.python_io.TFRecordWriter(output_file_names[i]) as writer:
                with tf.Session() as sess:
                    while True:
                        try:
                            batch_dict = OrderedDict()
                            batch_data = sess.run(iterator)
                            self._extract_id_labels(batch_dict,
                                                    batch_data,
                                                    self.output_labels_clean)
                            # For Each Record Get Split Assignment
                            # and add to current split if match
                            for ii, idd in enumerate(batch_dict.keys()):
                                if (idd in id_to_split_assignments) and \
                                   (id_to_split_assignments[idd] == split):
                                    record_dict = dict()
                                    record_dict['id'] = idd
                                    record_dict['labels'] = id_label_dict[idd]
                                    record_dict['images'] = \
                                        list(batch_data['images'][ii])
                                    serialized = self.tfr_encoder(
                                        record_dict,
                                        labels_are_numeric=True)
                                    writer.write(serialized)
                                else:
                                    continue
                        except tf.errors.OutOfRangeError:
                            break

    def _extract_id_labels(self, dict_all, data_batch, output_labels):
        """ Extract ids and labels from dataset and add to dict
            {'1234': {'labels/primary': ['cat', 'dog']}}
        """
        for i, idd in enumerate(list(data_batch['id'])):
            id_clean = str(idd, 'utf-8')
            dict_all[id_clean] = dict()
            for lab in output_labels:
                lab_i = data_batch[lab][i]
                lab_i = [str(x, 'utf-8') for x in lab_i]
                dict_all[id_clean][lab] = lab_i

    def _convert_id_label_dict_to_inventory(self, id_label_dict):
        """ convert id label dict to inventory """
        data_inv = DatasetInventory()
        data_inv.data_inventory = OrderedDict()
        inv = data_inv.data_inventory
        for record_id, label_types in id_label_dict.items():
            inv[record_id] = {'labels': dict()}
            for label_type, label_list in label_types.items():
                inv[record_id]['labels'][label_type] = label_list
        data_inv.label_handler = LabelHandler(data_inv.data_inventory)
        return data_inv

    def _convert_inventory_to_id_label_dict(self, inventory):
        """ convert dataset inventory to label dict """
        id_label_dict = OrderedDict()
        for record_id, data in inventory.data_inventory.items():
            id_label_dict[record_id] = dict()
            for label_type, labels in data['labels'].items():
                id_label_dict[record_id][label_type] = labels
        return id_label_dict

    def _convert_to_list(self, input):
        """ Convert input to list if str, else raise error """
        if isinstance(input, list):
            return input
        elif isinstance(input, str):
            return [input]
        else:
            raise ValueError("Function input: %s has to be a list, is: %s"
                             % (input, type(input)))

    # assign each id to a set
    def _assign_id_to_set(self, id_label_dict,
                          split_names,
                          split_props,
                          balanced_sampling_min=False,
                          balanced_sampling_label_type=None):
        """ Assign ID to Set """

        # assign unique hash to each id
        split_vals = {x: id_to_zero_one(x) for x in id_label_dict.keys()}

        # assign each id into different splits based on split value
        split_assignments = list()

        split_props_cum = [sum(split_props[0:(i+1)]) for i in
                           range(0, len(split_props))]

        for record_id in id_label_dict.keys():
            for sn, sp in zip(split_names, split_props_cum):
                split_val = split_vals[record_id]
                if split_val <= sp:
                    split_assignments.append(sn)
                    break

        # Balanced sampling to the minority class
        if balanced_sampling_min:
            if balanced_sampling_label_type is None:
                raise ValueError("balanced_sampling_label_type must not \
                                  be None if balanced_sampling_min = True")

            # count number of records for each label of relevant label type
            # To identify label with minimum number of records
            label_stats = dict()
            for v in id_label_dict.values():
                for label_type, labels in v.items():
                    if label_type == balanced_sampling_label_type:
                        for label in labels:
                            if label not in label_stats:
                                label_stats[label] = 0
                            label_stats[label] += 1

            # find minimum label
            min_label = min(label_stats, key=label_stats.get)
            min_value = label_stats[min_label]

            # assign each id to one unique class
            class_assignment = {x: list() for x in label_stats.keys()}
            remaining_record_ids = set()

            # Randomly Shuffle Ids
            all_record_ids = list(id_label_dict.keys())
            shuffle(all_record_ids)

            for record_id in all_record_ids:
                label_types = id_label_dict[record_id]
                label = label_types[balanced_sampling_label_type][0]
                # Add record to class assignment if label occurrence
                # is below min_value of least frequent class
                if len(class_assignment[label]) < min_value:
                    class_assignment[label].append(record_id)
                    remaining_record_ids.add(record_id)

        else:
            remaining_record_ids = id_label_dict.keys()

        # create final dictionary with split assignment per record id
        final_split_assignments = dict()

        for record_id, sp in zip(id_label_dict.keys(), split_assignments):
            if record_id in remaining_record_ids:
                final_split_assignments[record_id] = sp

        return final_split_assignments
