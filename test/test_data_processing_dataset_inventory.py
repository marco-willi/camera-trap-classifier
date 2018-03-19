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

    def testRemoveLabelTypes(self):
        dat = self.dinv.get_record_id_data("10300569")
        self.assertIn('color', dat['labels'])
        self.dinv.remove_label_types(["color"])
        dat = self.dinv.get_record_id_data("10300569")
        self.assertNotIn('color', dat['labels'])
        self.assertIn('primary', dat['labels'])

    def testRemoveLabelIfOneOfMany(self):
        self.dinv.remove_labels({'primary': 'human'})
        dat = self.dinv.get_record_id_data("multi_labels")
        self.assertNotIn('human', set(dat['labels']['primary']))

    def testRemoveLabelsIfOnlyLabel(self):
        self.dinv.remove_labels({'primary': ['elephant']})
        self.assertNotIn("is_elephant", self.inventory)

    def testKeepOnlyLabelTypes(self):
        self.dinv.keep_only_label_types(['color'])
        self.assertNotIn('primary', self.inventory["10300788"]['labels'])

    def testKeepOnlyLabelsAndRemoveOthers(self):
        self.dinv.keep_only_labels({"primary": ["elephant"]})
        self.assertIn('is_elephant', self.inventory)
        self.assertEqual(1, len(self.inventory.keys()))

    def testKeepOnlyLabels(self):
        self.dinv.keep_only_labels({"primary": ["dog", "cat"]})
        self.assertIn("multi_labels", self.inventory)
        self.assertNotIn('is_elephant', self.inventory)

