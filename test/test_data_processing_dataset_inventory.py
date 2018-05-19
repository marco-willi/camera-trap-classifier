import unittest
from data_processing.data_inventory import DatasetInventory


class DataInventoryTests(unittest.TestCase):
    """ Test Creation of Dataset Inventory """

    def setUp(self):
        self.dinv = DatasetInventory()
        self.path = './test/test_files/json_data_file.json'
        self.dinv.create_from_source(type='json', path=self.path)
        self.inventory = self.dinv.data_inventory

    def testRemoveRecord(self):
        self.dinv.remove_record("10300652")
        self.assertNotIn("10300652",  self.inventory)

    def testRemoveNotAllLabelTypes(self):
        self.assertNotIn("not_all_labels", self.inventory)
