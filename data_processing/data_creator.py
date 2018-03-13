""" Class To Create Dataset Inventory """
from config.config import logging
import os
from data_processing.utils import clean_input_path
import json


class DatasetInventory(object):
    """ Creates Datset Dictionary - Contains labels, links and data about each
        Record
    """
    def __init__(self):
        self.dataset_dict = None

    def get_dataset_inventory(self):
        return self.dataset_dict

    def get_all_record_ids(self):
        """ Get all ids of the inventory """
        return list(self.dataset_dict.keys())

    def get_all_label_types(self):
        """ get all label types in the dataset """
        all_label_types = set()
        for v in self.dataset_dict.values():
            all_label_types = all_label_types.union(v['labels'].keys())
        return all_label_types

    def get_record_id_data(self, record_id):
        """ Get content of record id """
        return self.dataset_dict[record_id]

    def get_number_of_records(self):
        """ Count and Return number of records """
        return len(self.dataset_dict.keys())

    def get_all_labels(self):
        """ get all label types and classes """
        all_labels = dict()
        for v in self.dataset_dict.values():
            for cl_type, cl_list in v['labels'].items():
                if cl_type not in all_labels:
                    all_labels[cl_type] = dict()
                for cl in cl_list:
                    if cl not in all_labels[cl_type]:
                        all_labels[cl_type][cl] = 0
                    all_labels[cl_type][cl] += 1
        return all_labels

    def _find_multi_label_records(self):
        """ Find records with mutliple entries in one label type """
        ids_to_remove = set()
        for record_id, record_data in self.dataset_dict.values():
            for label_type, labels in record_data['labels'].items():
                if len(labels) > 0:
                    ids_to_remove.add(record_id)
        return ids_to_remove

    def remove_multi_label_records(self):
        """ Removes records with mutliple entries in one label type """
        ids_to_remove = self._find_multi_label_records()
        for id_to_remove in ids_to_remove:
            self.remove_record(id_to_remove)

    def remove_record(self, id_to_remove):
        """ Remove specific record """
        self.dataset_dict.pop(id_to_remove, None)

    def create_from_class_directories(self, root_path):
        """ Create inventory from path which contains class-specific
            directories
        """
        root_path = clean_input_path(root_path)
        assert os.path.exists(root_path), "Path: %s does not exist" % root_path

        all_records_dict = self._create_dict_from_image_folders(root_path)
        self.dataset_dict = all_records_dict

    def _create_dict_from_image_folders(self, root_path):
        """ create dictionary from image paths """

        class_dir_name_list = os.listdir(root_path)

        self.n_classes = len(class_dir_name_list)
        logging.info("Found %s classes" % self.n_classes)
        logging.debug("Found following classes %s" % class_dir_name_list)

        # Process each image and create data dictionary
        all_images_data = dict()
        for class_dir in class_dir_name_list:
            for image_name in os.listdir(root_path + class_dir):
                splitted_file_name = image_name.split(".")
                if len(splitted_file_name) > 2:
                    raise ValueError("File %s has more than one . " +
                                     "in filename, which is not allowed"
                                     % str(image_name))
                unique_image_id = splitted_file_name[0]
                image_data = {
                    'images': [root_path + class_dir +
                               os.path.sep + image_name],
                    'labels': {'primary': [class_dir]}}
                all_images_data[unique_image_id] = image_data

        logging.info("Found %s images" % len(all_images_data.keys()))

        return all_images_data


class CamCatDatasetInventory(DatasetInventory):
    def create_from_json(self, path_to_json):
        """ Create Iventory from Json """
        all_records_dict = self._read_and_check_json(path_to_json)
        self.dataset_dict = all_records_dict

    def _read_and_check_json(self, path_to_json):
        """ Read Json File and Check Format """

        assert os.path.exists(path_to_json), \
            "Path: %s does not exist" % path_to_json

        try:
            data_dict = json.load(open(path_to_json))
        except Exception as e:
            logging.error('Failed to read Json:\n' + str(e))
            raise

        n_records = len(data_dict.keys())
        logging.info("Found %s records in %s" % (n_records, path_to_json))

        # Calculate and log statistics about labels
        label_stats = dict()
        for k, v in data_dict.items():
            if "labels" not in v:
                logging.info("Record %s has no 'labels' attr and is removed" %
                             k)
            # For each record get and count label types and labels
            for label_type, label_list in v['labels'].items():
                if type(label_list) is not list:
                    if type(label_list) is str:
                        label_list = [label_list]
                        data_dict[k]['labels'][label_type] = label_list
                    else:
                        logging.info("Record %s has invalid label type: %s" %
                                     (k, type(label_list)))
                        raise ImportError("Record has invalid label type")

                if label_type not in label_stats:
                    label_stats[label_type] = dict()
                for label in label_list:
                    if label not in label_stats[label_type]:
                        label_stats[label_type][label] = 0
                    label_stats[label_type][label] += 1

        # Log stats
        for k, v in label_stats.items():
            for label, label_count in v.items():
                logging.info("Label Type: %s - %s records for %s" %
                             (k, label_count, label))

        # check that each record has each label type
        all_label_types = label_stats.keys()
        multi_labels_per_type = dict()

        for k, v in data_dict.items():
            if not (v['labels'].keys() >= all_label_types) and \
                    (v['labels'].keys() <= all_label_types):
                logging.info("Record %s has not all label types: %s" %
                             (k, all_label_types))
                raise ValueError("Record without all required label types")

            # Count how many records have multiple labels per label type
            for label_type in v['labels']:
                if label_type not in multi_labels_per_type:
                    multi_labels_per_type[label_type] = 0
                multi_labels_per_type[label_type] += (len(label_type) > 0)

        for k, v in multi_labels_per_type.items():
            logging.info("Label Type %s has %s records with multiple labels" %
                         (k, v))

        return data_dict
