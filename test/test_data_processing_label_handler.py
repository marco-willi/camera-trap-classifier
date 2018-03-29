import unittest
from data_processing.data_inventory import DatasetInventory
from data_processing.label_handler import LabelHandler


#dinv = DatasetInventory()
#test_path_json = './test/test_files/json_data_file.json'
#dinv.create_from_json(test_path_json)
#inventory = dinv.data_inventory
#
#
#
#dinv.label_handler.keep_only_labels({"primary": ["elephant"], "color": ['brown', 'black']})
#assertIn('is_elephant', inventory)
#assertEqual(1, len(inventory.keys()))


class LabelHandlerTests(unittest.TestCase):
    """ Test Creation of Dataset Inventory """

    def setUp(self):
        self.dinv = DatasetInventory()
        self.test_path_json = './test/test_files/json_data_file.json'
        self.dinv.create_from_json(self.test_path_json)
        self.inventory = self.dinv.data_inventory

    def testRemoveMultipleLabels(self):
        self.dinv.label_handler.remove_multi_label_records()
        self.assertNotIn("multi_labels", self.inventory)

    def testGetAllLabelTypes(self):
        self.assertEqual({'color', 'primary'},
                         self.dinv.label_handler.get_all_label_types())

    def testRemoveLabelTypes(self):
        dat = self.dinv.get_record_id_data("10300569")
        self.assertIn('color', dat['labels'])
        self.dinv.label_handler.remove_label_types(["color"])
        dat = self.dinv.get_record_id_data("10300569")
        self.assertNotIn('color', dat['labels'])
        self.assertIn('primary', dat['labels'])

    def testRemoveLabelIfOneOfMany(self):
        self.dinv.label_handler.remove_labels({'primary': 'human'})
        dat = self.dinv.get_record_id_data("multi_labels")
        self.assertNotIn('human', set(dat['labels']['primary']))

    def testRemoveLabelsIfOnlyLabel(self):
        self.dinv.label_handler.remove_labels({'primary': ['elephant']})
        self.assertNotIn("is_elephant", self.inventory)

    def testKeepOnlyLabelTypes(self):
        self.dinv.label_handler.keep_only_label_types(['color'])
        self.assertNotIn('primary', self.inventory["10300788"]['labels'])

    def testKeepOnlyLabelsAndRemoveOthers(self):
        self.dinv.label_handler.keep_only_labels({"primary": ["elephant"]})
        self.assertIn('is_elephant', self.inventory)
        self.assertEqual(1, len(self.inventory.keys()))

    def testKeepOnlyLabelsAndRemoveOthersMulti(self):
        self.dinv.label_handler.keep_only_labels({"primary": ["elephant"], "color": ['brown', 'black']})
        self.assertIn('is_elephant', self.inventory)
        self.assertEqual(1, len(self.inventory.keys()))

    def testKeepOnlyLabels(self):
        self.dinv.label_handler.keep_only_labels({"primary": ["dog", "cat"]})
        self.assertIn("multi_labels", self.inventory)
        self.assertNotIn('is_elephant', self.inventory)

    def testKeepOnlyLabelsNotInLabelList(self):
        self.dinv.label_handler.keep_only_labels({"primary": ["dog", "cat", "genet"]})
        self.assertIn("multi_labels", self.inventory)
        self.assertNotIn('is_elephant', self.inventory)

    def testMapLabelsStandard(self):

        map1 = {'primary': {'cat': 'elephant', 'dog': 'giraffe'}}
        ex1 = {"cat_to_elephant": {'labels': {"primary": ['cat']}},
               "dog_to_giraffe": {'labels': {"primary": ['dog']}}}


        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': {"primary": ['elephant']}})
        self.assertEqual(res["dog_to_giraffe"], {'labels': {"primary": ['giraffe']}})

    def testMapLabelsManyToFew(self):
        map1 = {'primary': {'cat': 'elephant', 'dog': 'giraffe',
                            'lion': 'elephant'}}

        ex1 = {"cat_to_elephant": {'labels': {"primary": ['cat']}},
               "dog_to_giraffe": {'labels': {"primary": ['dog']}},
               "lion_to_elephant": {'labels': {"primary": ['lion']}}}

        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': {"primary": ['elephant']}})
        self.assertEqual(res["dog_to_giraffe"], {'labels': {"primary": ['giraffe']}})
        self.assertEqual(res["lion_to_elephant"], {'labels': {"primary": ['elephant']}})

    def testMapLabelsIfMissingInMapper(self):
        map1 = {'primary': {'cat': 'elephant'}}
        ex1 = {"cat_to_elephant": {'labels': {"primary": ['cat']}},
               "dog_to_dog": {'labels': {"primary": ['dog']}}}

        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': {"primary": ['elephant']}})
        self.assertEqual(res["dog_to_dog"], {'labels': {"primary": ['dog']}})

    def testMapIfLabelTypeMissingInMapper(self):
        map1 = {'primary': {'cat': 'elephant'}}
        ex1 = {"cat_to_elephant": {'labels': {"primary": ['cat']}},
               "dog_to_white": {'labels': {"color": ['white']}}}

        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': {"primary": ['elephant']}})
        self.assertEqual(res["dog_to_white"], {'labels': {"color": ['white']}})

    def testMapClassesToNumericStandard(self):
        ex1 = {"cat_to_0": {'labels': {"primary": ['cat']}},
               "dog_to_1": {'labels': {"primary": ['dog']}}}

        label_handler = LabelHandler(ex1)
        label_handler._create_mapping_labels_to_numeric()
        res = label_handler.labels_to_numeric

        self.assertEqual(res, {'primary': {'cat': 0, 'dog': 1}})

    def testMapClassesToNumericMulti(self):
        ex1 = {"cat_to_0": {'labels': {"primary": ['cat', 'elephant']}},
               "dog_to_1": {'labels': {"primary": ['dog'],
                            'color': ['purple', 'green']}}}

        label_handler = LabelHandler(ex1)
        label_handler._create_mapping_labels_to_numeric()
        res = label_handler.labels_to_numeric

        self.assertEqual(res['primary'],  {'cat': 0, 'dog': 1, 'elephant': 2})
        self.assertEqual(res['color'],  {'green': 0, 'purple': 1})
