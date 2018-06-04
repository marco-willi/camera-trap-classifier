import unittest
from data_processing.data_inventory import DatasetInventory


 # path = './test/test_files/json_data_file.json'
 # source_type = 'json'
 # dinv = DatasetInventory()
 # dinv.create_from_source(source_type, path)
 # inventory = dinv.data_inventory
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


class DataInventoryTests(unittest.TestCase):
    """ Test Creation of Dataset Inventory """

    def setUp(self):
        path = './test/test_files/json_data_file.json'
        source_type = 'json'
        params = {'path': path}
        self.dinv = DatasetInventory()
        self.dinv.create_from_source(source_type, params)
        self.inventory = self.dinv.data_inventory

    def testRemoveRecord(self):
        self.dinv.remove_record("10300652")
        self.assertNotIn("10300652",  self.inventory)

    def testRemoveNotAllLabelTypes(self):
        self.assertNotIn("not_all_labels", self.inventory)

#    def testTFRecordFormat(self):
#         self.dinv._get_tfr_record_format('single_species_standard')
#         self.dinv._get_tfr_record_format('single_species_multi_color')
#         self.dinv._get_tfr_record_format('multi_species_standard')
