import unittest
from data.inventory import (
    DatasetInventory, DatasetInventoryMaster)


class DataInventoryTests(unittest.TestCase):
    """ Test Creation of Dataset Inventory """

    def setUp(self):
        path = './test/test_files/json_data_file.json'
        source_type = 'json'
        params = {'path': path}
        self.dinv = DatasetInventoryMaster()
        self.dinv.create_from_source(source_type, params)
        self.inventory = self.dinv.data_inventory

    def testRemoveRecord(self):
        self.assertIn("single_species_standard",  self.inventory)
        self.dinv.remove_record("single_species_standard")
        self.assertNotIn("single_species_standard",  self.inventory)

    def testRemoveRecordsWithLabel(self):
        label_names = ['class', 'counts']
        label_values = ['elephant', '12']
        self.assertIn("is_elephant",  self.inventory)
        self.assertIn("counts_is_12",  self.inventory)
        self.dinv.remove_records_with_label(label_names, label_values)
        self.assertNotIn("is_elephant",  self.inventory)
        self.assertNotIn("counts_is_12",  self.inventory)

    def testKeepOnlyRecordsWithLabel(self):
        label_names = ['class', 'counts']
        label_values = ['elephant', '12']
        self.assertIn("is_elephant",  self.inventory)
        self.assertIn("single_species_standard",  self.inventory)
        self.assertIn("counts_is_12",  self.inventory)
        self.dinv.keep_only_records_with_label(label_names, label_values)
        self.assertNotIn("single_species_standard",  self.inventory)
        self.assertIn("is_elephant",  self.inventory)
        self.assertIn("counts_is_12",  self.inventory)

    def testConvertToTFRecordFormat(self):
        id = 'single_species_standard'
        self.dinv._map_labels_to_numeric()
        record = self.inventory[id]
        tfr_dict = self.dinv._convert_record_to_tfr_format(id, record)
        self.assertEqual(tfr_dict['id'], 'single_species_standard')
        self.assertEqual(tfr_dict['n_images'], 3)
        self.assertEqual(tfr_dict["image_paths"],
         ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
          "\\images\\4715\\all\\cat\\10296726_0.jpeg",
          "\\images\\4715\\all\\cat\\10296727_0.jpeg"])
        self.assertIsInstance(tfr_dict["label_num/class"][0], int)
        self.assertEqual(tfr_dict["label_num/color_brown"], [1])
        self.assertEqual(tfr_dict["label_num/color_white"], [0])
        self.assertIsInstance(tfr_dict["label_num/counts"][0], int)
        self.assertEqual(tfr_dict["label/class"], ['cat'])
        self.assertEqual(tfr_dict["label/color_brown"], ['1'])
        self.assertEqual(tfr_dict["label/color_white"], ['0'])
        self.assertEqual(tfr_dict["label/counts"], ['1'])


#    def testTFRecordFormat(self):
#         self.dinv._get_tfr_record_format('single_species_standard')
#         self.dinv._get_tfr_record_format('single_species_multi_color')
#         self.dinv._get_tfr_record_format('multi_species_standard')

#path = './test_big/cats_vs_dogs/inventory.json'
#source_type = 'json'
#params = {'path': path}
#dinv = DatasetInventoryMaster()
#dinv.create_from_source(source_type, params)
#inventory = dinv.data_inventory
#dinv.keep_only_records_with_label(['class', 'class'], ['cat', 'dog'])
#
#tt = dinv.split_inventory_by_random_splits_with_balanced_sample(
#            split_label_min='class',
#            split_names=['test', 'train'],
#            split_percent=[0.5 ,0.5])
#


# dinv._get_tfr_record_format('single_species_standard')
# dinv._get_tfr_record_format('single_species_multi_color')
# dinv._get_tfr_record_format('multi_species_standard')
#
#
#
#
# tt = dinv.label_handler.get_all_label_types()
# tt = dinv.label_handler._find_all_label_attributes()
# dinv.label_handler.get_all_labels()
# dinv.label_handler._create_mapping_labels_to_numeric()
# dinv.label_handler.labels_to_numeric
#
#
# record_data = inventory['single_species_standard']
#
# labels_list = dinv.label_handler._extract_labels_from_record(record_data)
# label_types = dinv.label_handler._extract_label_types_from_labels_list(labels_list)
# dinv.label_handler._find_all_label_attributes()
#
# record_data = list(dinv.label_handler.inv_data.values())[0]
# label_entries = set()
# label_entries.union(label_types)
#
# path = "D:\Studium_GD\Zooniverse\CamCatProject\data\inventory_list.csv"
# source_type = 'panthera_csv'
# dinv = DatasetInventory()
# dinv.create_from_source(source_type, path)
# inventory = dinv.data_inventory
