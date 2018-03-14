""" Class To Create Dataset Inventory """
from config.config import logging
from data_processing.data_importer import ImportFromJson, ImportFromImageDirs


class DatasetInventory(object):
    """ Creates Datset Dictionary - Contains labels, links and data about each
        Record
    """
    def __init__(self):
        self.data_inventory = None

    def get_all_record_ids(self):
        """ Get all ids of the inventory """
        return list(self.data_inventory.keys())

    def get_all_label_types(self):
        """ get all label types in the dataset """
        all_label_types = set()
        for v in self.data_inventory.values():
            all_label_types = all_label_types.union(v['labels'].keys())
        return all_label_types

    def get_record_id_data(self, record_id):
        """ Get content of record id """
        return self.data_inventory[record_id]

    def get_number_of_records(self):
        """ Count and Return number of records """
        return len(self.data_inventory.keys())

    def get_all_labels(self):
        """ get all label types and classes """
        all_labels = dict()
        for v in self.data_inventory.values():
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
        for record_id, record_data in self.data_inventory.items():
            for labels in record_data['labels'].values():
                if len(labels) > 1:
                    ids_to_remove.add(record_id)
        return ids_to_remove

    def remove_multi_label_records(self):
        """ Removes records with mutliple entries in one label type """
        ids_to_remove = self._find_multi_label_records()
        for id_to_remove in ids_to_remove:
            self.remove_record(id_to_remove)

    def remove_record(self, id_to_remove):
        """ Remove specific record """
        self.data_inventory.pop(id_to_remove, None)

    def create_from_class_directories(self, root_path):
        """ Create inventory from path which contains class-specific
            directories
        """
        class_dir_reader = ImportFromImageDirs()
        self.data_inventory = \
            class_dir_reader.read_from_image_root_dir(root_path)

        self._remove_not_all_label_types_present()

    def create_from_json(self, json_path):
        """ Create inventory from json file """
        json_reader = ImportFromJson()
        self.data_inventory = \
            json_reader.read_from_json(json_path)

        self._remove_not_all_label_types_present()

    def _remove_not_all_label_types_present(self):
        """ Remove Records which dont have all labels """

        # get all label types in the data inventory
        all_label_types = set()
        for record_data in self.data_inventory.values():
            for label_type in record_data['labels']:
                all_label_types.add(label_type)

        # remove records which contain not all types
        ids_to_remove = list()
        for record_id, record_data in self.data_inventory.items():
            for label_type in all_label_types:
                if label_type not in record_data['labels']:
                    ids_to_remove.append(record_id)

        for record_to_remove in ids_to_remove:
            logging.info("Record %s has not all label types: %s" %
                         (record_to_remove, all_label_types))
            self.remove_record(record_to_remove)

    def log_stats(self):
        """ Logs Statistics about Data Inventory """

        # Calculate and log statistics about labels
        label_stats = dict()
        label_type_stats = dict()
        for k, v in self.data_inventory.items():
            # For each record get and count label types and labels
            for label_type, label_list in v['labels'].items():
                if label_type not in label_stats:
                    label_stats[label_type] = dict()
                    label_type_stats[label_type] = 0

                # Count if multiple labels
                if len(label_list) > 1:
                    label_type_stats[label_type] += 1

                for label in label_list:
                    if label not in label_stats[label_type]:
                        label_stats[label_type][label] = 0
                    label_stats[label_type][label] += 1

        # Log stats
        for k, v in label_stats.items():
            for label, label_count in v.items():
                logging.info("Label Type: %s - %s records for %s" %
                             (k, label_count, label))

        # Multiple Labels per Label Type
        for k, v in label_type_stats.items():
            logging.info("Label Type %s has %s records with multiple labels" %
                         (k, v))
