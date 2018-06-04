import unittest
from data_processing.data_inventory import DatasetInventory
from data_processing.label_handler import LabelHandler


# path = './test/test_files/json_data_file.json'
# source_type = 'json'
# dinv = DatasetInventory()
# dinv.create_from_source(source_type, path)
# inventory = dinv.data_inventory
#
#
# self.dinv.label_handler.remove_labels({'class': 'dog'})
# dat = self.dinv.get_record_id_data("multi_species_standard")
# classes = [x['class'] for x in dat['labels']]

# to_remove = dinv.label_handler.get_all_labels()
# label_type_labels = {"class": ["dog", "cat"]}
#
# for label_type, labels in to_remove.items():
#     if label_type not in label_type_labels:
#         to_remove[label_type] = list(labels)
#     else:
#         labels_to_remove = set(to_remove[label_type].keys())
#         for label in label_type_labels[label_type]:
#             if label in labels_to_remove:
#                 labels_to_remove.remove(label)
#         to_remove[label_type] = list(labels_to_remove)
#
#
# # convert all label entries to list
# label_type_labels = to_remove
# for label_type, labels in label_type_labels.items():
#     label_type_labels[label_type] = dinv.label_handler._convert_to_list(labels)
#
# new_record_label = dict()
#
#
# labels = dinv.label_handler._extract_labels_from_record(dinv.data_inventory["is_elephant"])
# new_labels = copy.deepcopy(labels)
#
# for i, label in enumerate(labels):
#     for label_attr, label_vals in label_type_labels.items():
#         if label_attr in label.keys():
#             for label_val in label_vals:
#                 if label_val in label[label_attr]:
#                     if isinstance(new_labels[i][label_attr], list):
#                         new_labels[i][label_attr].remove(label_val)
#                     else:
#                         if new_labels[i][label_attr] == label_val:
#                             new_labels[i].pop(label_attr, None)
#
#
#
#
# dinv.label_handler.keep_only_labels({"class": ["dog", "cat"]})


#
# dinv.label_handler._create_mapping_labels_to_numeric()
# dinv.label_handler.get_all_labels()
#
#
# tt = dinv.label_handler._find_all_label_attributes()
# dinv.label_handler._extract_label_types_from_labels_list()
#
#
# all_ids = dinv.get_all_record_ids()
# dinv.label_handler.remove_multi_label_records()
# no_multi_ids = set(all_ids) - set(["multi_species_standard",
#                               "multi_species_no_other_attr"])
#
# set(dinv.get_all_record_ids())
#
# dinv.label_handler.get_all_label_types()
#
# dinv.label_handler.remove_label_types(["color"])
#
#
#map1 = {'class': {'cat': 'elephant'}}
# map1 = {'class': {'cat': 'elephant'}}
# ex1 = {"cat_to_elephant": {'labels': [{"class": ['cat']}]},
#        "dog_to_white": {'labels': [{"color": ['white']}]}}
#
# label_handler = LabelHandler(ex1)
#
# all_labels = {k: dict() for k in label_handler._find_all_label_attributes()}
#
# for record_info in label_handler.inv_data.values():
#     labels = label_handler._extract_labels_from_record(record_info)
#     for label in labels:
#         for label_name, label_vals in label.items():
#             if isinstance(label_vals, list):
#                 for label_val in label_vals:
#                     if label_val not in all_labels[label_name]:
#                         all_labels[label_name][label_val] = 0
#                     all_labels[label_name][label_val] += 1
#             else:
#                 if label_vals not in all_labels[label_name]:
#                     all_labels[label_name][label_vals] = 0
#                 all_labels[label_name][label_vals] += 1
#
#
#
# label_handler.get_all_label_types()
# label_handler.get_all_labels()
#
# label_handler._create_mapping_labels_to_numeric()
# label_handler.labels_to_numeric
#
# label_handler.map_labels(map1)
#
#
# res = label_handler.inv_data
#
#
# dinv.label_handler.labels_to_numeric
#
#
# record_data = inventory['single_species_standard']


#dinv.label_handler.keep_only_labels({"primary": ["elephant"], "color": ['brown', 'black']})
#assertIn('is_elephant', inventory)
#assertEqual(1, len(inventory.keys()))


class LabelHandlerTests(unittest.TestCase):
    """ Test Creation of Dataset Inventory """

    def setUp(self):
        path = './test/test_files/json_data_file.json'
        source_type = 'json'
        params = {'path': path}
        self.dinv = DatasetInventory()
        self.dinv.create_from_source(source_type, params)
        self.inventory = self.dinv.data_inventory

    def testRemoveMultipleLabels(self):
        all_ids = self.dinv.get_all_record_ids()
        self.dinv.label_handler.remove_multi_label_records()
        no_multi_ids = set(all_ids) - set(["multi_species_standard",
                                      "multi_species_no_other_attr"])
        self.assertEqual(no_multi_ids, set(self.dinv.get_all_record_ids()))


    def testGetAllLabelTypes(self):
        self.assertEqual({'class', 'color', 'counts'},
                         self.dinv.label_handler.get_all_label_types())

    def testRemoveLabelTypes(self):
        dat = self.dinv.get_record_id_data("single_species_standard")
        self.assertIn('color', dat['labels'][0])
        self.dinv.label_handler.remove_label_types(["color"])
        dat = self.dinv.get_record_id_data("single_species_standard")
        self.assertNotIn('color', dat['labels'][0])
        self.assertIn('counts', dat['labels'][0])

    def testRemoveLabelIfOneOfMany(self):
        self.dinv.label_handler.remove_labels({'class': 'dog'})
        dat = self.dinv.get_record_id_data("multi_species_standard")
        classes = [x['class'] for x in dat['labels']]
        self.assertNotIn('dog', classes)
        self.assertIn('cat', classes)

    def testRemoveLabelsIfOnlyLabel(self):
        self.dinv.label_handler.remove_labels({'class': ['elephant']})
        self.assertNotIn("is_elephant", self.inventory)

    def testKeepOnlyLabelTypes(self):
        self.dinv.label_handler.keep_only_label_types(['color'])
        self.assertNotIn('class', self.inventory["single_species_standard"]['labels'][0])

    def testKeepOnlyLabelsAndRemoveOthers(self):
        self.dinv.label_handler.keep_only_labels({"class": ["elephant"]})
        self.assertIn('is_elephant', self.inventory)
        self.assertEqual(1, len(self.inventory.keys()))

    def testKeepOnlyLabelsAndRemoveOthersMulti(self):
        self.dinv.label_handler.keep_only_labels(
            {"class": ["elephant"],
             "color": ['gray', 'black'], "counts": ["1", "2", "3"]})
        self.assertIn('is_elephant', self.inventory)
        self.assertEqual(1, len(self.inventory.keys()))

    def testKeepOnlyLabels(self):
        self.dinv.label_handler.keep_only_labels({"class": ["dog", "cat"]})
        self.assertIn("multi_species_standard", self.inventory)
        self.assertNotIn('is_elephant', self.inventory)

    def testKeepOnlyLabelsNotInLabelList(self):
        self.dinv.label_handler.keep_only_labels({"class": ["dog", "cat", "genet"]})
        self.assertIn("multi_species_standard", self.inventory)
        self.assertNotIn('is_elephant', self.inventory)

    def testMapLabelsStandard(self):

        map1 = {'class': {'cat': 'elephant', 'dog': 'giraffe'}}
        ex1 = {"cat_to_elephant": {'labels': [{"class": ['cat']}]},
               "dog_to_giraffe": {'labels': [{"class": ['dog']}]}}


        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': [{"class": ['elephant']}]})
        self.assertEqual(res["dog_to_giraffe"], {'labels': [{"class": ['giraffe']}]})

    def testMapLabelsManyToFew(self):
        map1 = {'class': {'cat': 'elephant', 'dog': 'giraffe',
                            'lion': 'elephant'}}

        ex1 = {"cat_to_elephant": {'labels': [{"class": ['cat']}]},
               "dog_to_giraffe": {'labels': [{"class": ['dog']}]},
               "lion_to_elephant": {'labels': [{"class": ['lion']}]}}

        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': [{"class": ['elephant']}]})
        self.assertEqual(res["dog_to_giraffe"], {'labels': [{"class": ['giraffe']}]})
        self.assertEqual(res["lion_to_elephant"], {'labels': [{"class": ['elephant']}]})

    def testMapLabelsIfMissingInMapper(self):
        map1 = {'class': {'cat': 'elephant'}}
        ex1 = {"cat_to_elephant": {'labels': [{"class": ['cat']}]},
               "dog_to_dog": {'labels': [{"class": ['dog']}]}}

        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': [{"class": ['elephant']}]})
        self.assertEqual(res["dog_to_dog"], {'labels': [{"class": ['dog']}]})

    def testMapIfLabelTypeMissingInMapper(self):
        map1 = {'class': {'cat': 'elephant'}}
        ex1 = {"cat_to_elephant": {'labels': [{"class": ['cat']}]},
               "dog_to_white": {'labels': [{"color": ['white']}]}}

        label_handler = LabelHandler(ex1)
        label_handler.map_labels(map1)
        res = label_handler.inv_data

        self.assertEqual(res["cat_to_elephant"], {'labels': [{"class": ['elephant']}]})
        self.assertEqual(res["dog_to_white"], {'labels': [{"color": ['white']}]})

    def testMapClassesToNumericStandard(self):
        ex1 = {"cat_to_0": {'labels': [{"class": ['cat']}]},
               "dog_to_1": {'labels': [{"class": ['dog']}]}}

        label_handler = LabelHandler(ex1)
        label_handler._create_mapping_labels_to_numeric()
        res = label_handler.labels_to_numeric

        self.assertEqual(res, {'class': {'cat': 0, 'dog': 1}})

    def testMapClassesToNumericMulti(self):
        ex1 = {"cat_to_0": {'labels': [{"class": ['cat', 'elephant']}]},
               "dog_to_1": {'labels': [{"class": ['dog'],
                            'color': ['purple', 'green']}]}}

        label_handler = LabelHandler(ex1)
        label_handler._create_mapping_labels_to_numeric()
        res = label_handler.labels_to_numeric

        self.assertEqual(res['class'],  {'cat': 0, 'dog': 1, 'elephant': 2})
        self.assertEqual(res['color'],  {'green': 0, 'purple': 1})
