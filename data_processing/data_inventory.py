""" Class To Create Dataset Inventory """
import random

from config.config import logging
from data_processing.data_importer import (
    ImportFromJson, ImportFromImageDirs, ImportFromPantheraCSV)
from data_processing.label_handler import LabelHandler


class DatasetInventory(object):
    """ Creates Datset Dictionary - Contains labels, links and data about each
        Record
    """
    def __init__(self):
        self.data_inventory = None
        self.label_handler = None

    def randomly_remove_samples_to_percent(self, p_keep):
        """ Randomly sample a percentage of all records """
        if not p_keep <= 1:
            raise ValueError("p has to be between 0 and 1")

        new_data_inv = dict()
        all_ids = list(self.data_inventory.keys())
        n_total = len(all_ids)
        n_choices = int(n_total * p_keep)
        choices = random.choices(all_ids, k=n_choices)

        for id in choices:
            new_data_inv = self.data_inventory[id]

        self.data_inventory = new_data_inv

    def get_all_record_ids(self):
        """ Get all ids of the inventory """
        return list(self.data_inventory.keys())

    def get_record_id_data(self, record_id):
        """ Get content of record id """
        return self.data_inventory[record_id]

    def get_number_of_records(self):
        """ Count and Return number of records """
        return len(self.data_inventory.keys())

    def remove_record(self, id_to_remove):
        """ Remove specific record """
        self.data_inventory.pop(id_to_remove, None)

    def create_from_panthera_csv(self, path_to_csv):
        """ Create Inventory from a csv file according to Panthera-style
            formats
        """
        csv_reader = ImportFromPantheraCSV()
        self.data_inventory = csv_reader.read_from_csv(path_to_csv)
        self.label_handler = LabelHandler(self.data_inventory)
        self.label_handler.remove_not_all_label_types_present()

    def create_from_class_directories(self, root_path):
        """ Create inventory from path which contains class-specific
            directories
        """
        class_dir_reader = ImportFromImageDirs()
        self.data_inventory = \
            class_dir_reader.read_from_image_root_dir(root_path)

        self.label_handler = LabelHandler(self.data_inventory)
        self.label_handler.remove_not_all_label_types_present()

    def create_from_json(self, json_path):
        """ Create inventory from json file """
        json_reader = ImportFromJson()
        self.data_inventory = \
            json_reader.read_from_json(json_path)

        self.label_handler = LabelHandler(self.data_inventory)
        self.label_handler.remove_not_all_label_types_present()

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

        # Log Stats
        for label_type, labels in label_stats.items():
            label_list = list()
            count_list = list()
            for label, count in labels.items():
                label_list.append(label)
                count_list.append(count)
            total_counts = sum(count_list)
            sort_index = sorted(range(len(count_list)), reverse=True,
                                key=lambda k: count_list[k])
            for idx in sort_index:
                logging.info(
                    "Label Type: %s Label: %s Records: %s / %s (%s %%)" %
                    (label_type, label_list[idx], count_list[idx],
                     total_counts,
                     round(100 * (count_list[idx]/total_counts), 4)))

        # for k, v in label_stats.items():
        #     for label, label_count in v.items():
        #         logging.info("Label Type: %s - %s records for %s" %
        #                      (k, label_count, label))

        # Multiple Labels per Label Type
        for k, v in label_type_stats.items():
            logging.info("Label Type %s has %s records with multiple labels" %
                         (k, v))
