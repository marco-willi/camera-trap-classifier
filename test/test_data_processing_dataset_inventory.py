import unittest
from data_processing.data_inventory import DatasetInventory


class DataInventoryTests(unittest.TestCase):
    """ Test Creation of Dataset Inventory """

    def setUp(self):
        self.dinv = DatasetInventory()
        self.test_path_json = './test/test_files/json_data_file.json'
        self.dinv.create_from_json(self.test_path_json)
        self.inventory = self.dinv.data_inventory

    def testRemoveNotAllLabelTypes(self):
        self.assertNotIn("not_all_labels", self.inventory)

    def testRemoveMultipleLabels(self):
        self.dinv.remove_multi_label_records()
        self.assertNotIn("multi_labels", self.inventory)

    def testRemoveRecord(self):
        self.dinv.remove_record("10300652")
        self.assertNotIn("10300652",  self.inventory)

    def testGetAllLabelTypes(self):
        self.assertEqual({'color', 'primary'}, self.dinv.get_all_label_types())
