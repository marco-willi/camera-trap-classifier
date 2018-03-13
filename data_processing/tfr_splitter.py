""" Class To Split TFRecords """
from config.config import logging
import tensorflow as tf
from data_processing.utils import id_to_zero_one, n_records_in_tfr
from data_processing.data_reader import DatasetReader
from collections import OrderedDict


class TFRecordSplitter(object):
    """ Splits TFRecord files """

    def __init__(self, main_file, tfr_encoder_decoder):
        self.main_file = main_file
        self.tfr_encoder_decoder = tfr_encoder_decoder
        self.split_files = None
        self.split_props = None
        self.split_names = None
        self.class_mapping = None

    def get_split_files(self):
        """ Get Path to Splitted Files """
        return self.split_files

    def get_split_names(self):
        """ Get names of splits """
        return self.split_names

    def get_splits_dict(self):
        """ Get Split-Paths and Split-Names """
        return {k: v for k, v in zip(self.split_names, self.split_files)}

    def get_class_mapping(self):
        """ Get Class Mapping """
        return self.class_mapping

    def print_record_numbers_per_file(self):
        """ print record numbers per file """
        for f, n in zip(self.split_files, self.split_names):
            logging.info("File: %s has %s records" % (n, n_records_in_tfr(f)))

    def get_record_numbers_per_file(self):
        """ get record numbers per file in dict """
        return {n: n_records_in_tfr(f) for f, n in
                zip(self.split_files, self.split_names)}

    def split_tfr_file(self, output_path, output_prefix,
                       splits, split_props, output_labels,
                       class_mapping=None):
        """ Split a TFR file according to split proportions """

        # Check Split Proportions
        assert sum(split_props) == 1, "sum of split_prop has to be 1, is: %s" \
            % sum(split_props)

        self.split_props = split_props

        # Check Split Names
        assert len(splits) == len(split_props), "Number of splits must be " + \
            "equal to number of split_prop"

        self.split_names = splits

        # Check Split Types
        assert type(splits) is list and type(split_props) is list, \
            "splits and split_prop must be lists"

        # Check Class Mapping
        self.class_mapping = class_mapping

        # Rename class label types
        if self.class_mapping is not None:
            for label_type in self.class_mapping.keys():
                self.class_mapping['labels/' + label_type] = \
                    self.class_mapping.pop(label_type)

        # Create Output File Names
        output_file_names = [output_path + output_prefix + '_' +
                             s + '.tfrecord' for s in splits]

        self.split_files = output_file_names

        output_labels_clean = ['labels/' + x for x in output_labels]

        # get all ids and their labels
        logging.debug("Getting ids and labels via encoder_decoder")

        dataset_reader = DatasetReader(
            self.tfr_encoder_decoder._decode_labels_and_images)

        iterator = dataset_reader.get_iterator(
             self.main_file, batch_size=128, is_train=False, n_repeats=1,
             output_labels=output_labels,
             buffer_size=2048,
             max_multi_label_number=None)

        id_label_dict = OrderedDict()
        with tf.Session() as sess:
            while True:
                try:
                    ids, image_labels = sess.run(iterator)
                    self._extract_id_labels(id_label_dict, ids,
                                            image_labels, output_labels_clean)
                except tf.errors.OutOfRangeError:
                    break

        # map labels if specified
        if self.class_mapping is not None:
            id_label_dict = self._map_labels(id_label_dict, self.class_mapping)

        # create label to numeric mapper
        label_to_numeric_mapper = self._map_labels_to_numeric(id_label_dict)
        self.label_to_numeric_mapper = label_to_numeric_mapper

        # change labels to numeric
        id_label_dict = self._map_labels(id_label_dict,
                                         label_to_numeric_mapper)

        # assign each id to a splitting value
        id_to_split_val = {x: id_to_zero_one(x) for x in id_label_dict.keys()}

        # Write TFrecord files for each split
        for i, split in enumerate(splits):
            split_p_lower = sum(split_props[0:i])
            split_p_upper = sum(split_props[0:i+1])

            # Read TFrecord file and iterate over it
            dataset_reader = DatasetReader(
                self.tfr_encoder_decoder._decode_labels_and_images)

            iterator = dataset_reader.get_iterator(
                 self.main_file, batch_size=128, is_train=False, n_repeats=1,
                 output_labels=output_labels,
                 buffer_size=2048,
                 max_multi_label_number=None)

            # Write Split File
            logging.debug("Start writing file %s" % output_file_names[i])
            with tf.python_io.TFRecordWriter(output_file_names[i]) as writer:
                with tf.Session() as sess:
                    while True:
                        try:
                            batch_dict = OrderedDict()
                            ids, image_labels = sess.run(iterator)

                            self._extract_id_labels(batch_dict, ids,
                                                    image_labels,
                                                    output_labels_clean)

                            for ii, idd in enumerate(batch_dict.keys()):
                                if self._between(
                                 id_to_split_val[idd],
                                 split_p_lower, split_p_upper):
                                    record_dict = dict()
                                    record_dict['id'] = idd
                                    record_dict['labels'] = id_label_dict[idd]
                                    record_dict['images'] = \
                                        list(image_labels['images'][ii])
                                    serialized = self.tfr_encoder_decoder.\
                                        serialize_split_tfr_record(record_dict)
                                    writer.write(serialized)
                                else:
                                    continue
                        except tf.errors.OutOfRangeError:
                            break



            # # iterate over input file
            # data_iterator = tf.python_io.tf_record_iterator(self.main_file)
            #
            # # Write Split File
            # logging.debug("Start writing file %s" % output_file_names[i])
            # with tf.python_io.TFRecordWriter(output_file_names[i]) as writer:
            #
            #     # iterate over all records of the main file
            #     for ii, data_record in enumerate(data_iterator):
            #
            #         # create a splitting id and check if record belongs to
            #         # this set
            #         split_id = split_assignment[ii]
            #         if (split_id >= split_p_lower) and \
            #            (split_id <= split_p_upper):
            #
            #             # deserialize record_data
            #             record_dict = self.tfr_encoder_decoder.decode_label_and_images_to_dict(
            #                             data_record,
            #                             output_labels_clean)
            #
            #             # change labels
            #             new_label_dict = dict()
            #
            #             for label_type in output_labels_clean:
            #                 new_labels = list()
            #                 for label in record_dict[label_type]:
            #                     mapped_label = \
            #                         label_to_numeric_mapper[label_type][label]
            #                     new_labels.append(mapped_label)
            #                 new_label_dict[label_type] = new_labels
            #
            #             record_dict['labels'] = new_label_dict
            #
            #             serialized = self.tfr_encoder_decoder.serialize_split_tfr_record(record_dict)
            #
            #             # Write the serialized data to the TFRecords file.
            #             writer.write(serialized)

    def _extract_id_labels(self, dict_all, ids, labels, output_labels):
        """ Extract ids and labels from dataset and add to dict"""
        for i, idd in enumerate(list(ids['id'])):
            id_clean = str(idd, 'utf-8')
            dict_all[id_clean] = dict()
            for lab in output_labels:
                lab_i = labels[lab][i]
                lab_i = [str(x, 'utf-8') for x in lab_i]
                dict_all[id_clean][lab] = lab_i

    def _between(self, value, lower, upper):
        """ Check if value is between lower and upper """
        return (value <= upper) and (value > lower)

    def _map_labels_to_numeric(self, records_info):
        """ Map Labels To Numerics """
        # Create inventory of all label types and labels
        # input data test: {'lalal': {'primary': ['cat'],
        #  'color':["brown", "white"]}, "1233":
        #  {"primary": ["cat", "dog"], 'color': ["purple"]}}
        # Output: {'color': {'brown': 0, 'purple': 1, 'white': 2},
        #    'primary': {'cat': 0, 'dog': 1}}

        label_dict = dict()
        for record_id, label_types in records_info.items():
            for label_type, label_list in label_types.items():
                if label_type not in label_dict:
                    label_dict[label_type] = dict()
                for label in label_list:
                    if label not in label_dict[label_type]:
                        label_dict[label_type][label] = dict()
                    label_dict[label_type][label] = ''

        # Map labels of each label type alphabetically
        for label_type, labels in label_dict.items():
            label_list = list(labels.keys())
            label_list.sort()
            for i, sorted_label in enumerate(label_list):
                label_dict[label_type][sorted_label] = i

        return label_dict

    def _map_labels(self, records_info, mapper):
        """ Map existing labels to new labels """
        # Example class mapping: {'primary': {'cat':'elephant', 'dog':
        #    'giraffe'},'color':{'white':'brown', 'purple': 'brown'}}
        # Example record: {'lalal': {'primary': ['cat'],
        #     'color':["brown", "white"]}, "1233":
        #      {"primary": ["cat", "dog"], 'color': ["purple"]}}
        # Example result: {'1233': {'color': ['brown'], 'primary':
        #     ['giraffe', 'elephant']}, 'lalal':
        #      {'color': ['brown'], 'primary': ['elephant']}}
        for record_id, label_types in records_info.items():
            for label_type, label_list in label_types.items():
                new_label_list = set()
                if label_type in mapper:
                    for label in label_list:
                        if label in mapper[label_type]:
                            mapped_label = \
                                mapper[label_type][label]
                            new_label_list.add(mapped_label)
                        else:
                            new_label_list.add(label)
                    records_info[record_id][label_type] = \
                        list(new_label_list)
        return records_info
