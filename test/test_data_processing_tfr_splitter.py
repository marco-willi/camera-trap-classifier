import unittest
from data_processing.tfr_splitter import TFRecordSplitter
from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder


class TFRecordSpliterTester(unittest.TestCase):
    """ Test TFR splitter """

    def setUp(self):
        self.enc_dec = DefaultTFRecordEncoderDecoder()
        self.output_path = './test/test_files/'
        self.splitter = TFRecordSplitter(
            files_to_split='./test/test_files/test.tfrecord',
            tfr_encoder=self.enc_dec.encode_record,
            tfr_decoder=self.enc_dec.decode_record
            )

    def testMapLabelsStandard(self):
        map1 = {'primary': {'cat': 'elephant', 'dog': 'giraffe'}}
        ex1 = {"cat_to_elephant": {"primary": ['cat']},
               "dog_to_giraffe": {"primary": ['dog']}}
        res = self.splitter._map_labels(ex1, map1)

        self.assertEqual(res["cat_to_elephant"], {"primary": ['elephant']})
        self.assertEqual(res["dog_to_giraffe"], {"primary": ['giraffe']})

    def testMapLabelsManyToFew(self):
        map1 = {'primary': {'cat': 'elephant', 'dog': 'giraffe',
                            'lion': 'elephant'}}

        ex1 = {"cat_to_elephant": {"primary": ['cat']},
               "dog_to_giraffe": {"primary": ['dog']},
               "lion_to_elephant": {"primary": ['lion']}}

        res = self.splitter._map_labels(ex1, map1)

        self.assertEqual(res["cat_to_elephant"], {"primary": ['elephant']})
        self.assertEqual(res["dog_to_giraffe"], {"primary": ['giraffe']})
        self.assertEqual(res["lion_to_elephant"], {"primary": ['elephant']})

    def testMapLabelsIfMissingInMapper(self):
        map1 = {'primary': {'cat': 'elephant'}}
        ex1 = {"cat_to_elephant": {"primary": ['cat']},
               "dog_to_dog": {"primary": ['dog']}}
        res = self.splitter._map_labels(ex1, map1)

        self.assertEqual(res["cat_to_elephant"], {"primary": ['elephant']})
        self.assertEqual(res["dog_to_dog"], {"primary": ['dog']})

    def testMapIfLabelTypeMissingInMapper(self):
        map1 = {'primary': {'cat': 'elephant'}}
        ex1 = {"cat_to_elephant": {"primary": ['cat']},
               "dog_to_white": {"color": ['white']}}
        res = self.splitter._map_labels(ex1, map1)

        self.assertEqual(res["cat_to_elephant"], {"primary": ['elephant']})
        self.assertEqual(res["dog_to_white"], {"color": ['white']})

    def testMapClassesToNumericStandard(self):
        ex1 = {"cat_to_0": {"primary": ['cat']},
               "dog_to_1": {"primary": ['dog']}}
        res = self.splitter._map_labels_to_numeric(ex1)
        self.assertEqual(res, {'primary': {'cat': 0, 'dog': 1}})

    def testMapClassesToNumericMulti(self):
        ex1 = {"cat_to_0": {"primary": ['cat', 'elephant']},
               "dog_to_1": {"primary": ['dog'],
                            'color': ['purple', 'green']}}

        res = self.splitter._map_labels_to_numeric(ex1)
        self.assertEqual(res['primary'],  {'cat': 0, 'dog': 1, 'elephant': 2})
        self.assertEqual(res['color'],  {'green': 0, 'purple': 1})

    def testClassAssignmentStandard(self):
        id_label_dict1 = {str(x): {"primary": ["cat"]} for
                          x in range(0, int(20e3))}
        id_label_dict2 = {str(x): {"primary": ["dog"]} for
                          x in range(int(20e3), int(25e3))}
        id_label_dict3 = {str(x): {"primary": ["ele"]} for
                          x in range(int(25e3), int(27e3))}
        id_label_dict = {**id_label_dict1, **id_label_dict2, **id_label_dict3}

        n_total = len(id_label_dict.keys())

        split_names = ["train", "test", "val"]
        split_props = [0.5, 0.3, 0.2]

        set_assignment = self.splitter._assign_id_to_set(
            id_label_dict,
            split_names=["train", "test", "val"],
            split_props=[0.5, 0.3, 0.2],
            balanced_sampling_min=False,
            balanced_sampling_label_type=None
        )

        stats = {x: 0 for x in split_names}
        for k, v in set_assignment.items():
            self.assertIn(v, set(split_names))
            stats[v] += 1

        self.assertAlmostEquals(stats['train'], n_total * split_props[0],
         delta=n_total*0.05)
        self.assertAlmostEqual(stats['test'], n_total * split_props[1],
         delta=n_total*0.05)
        self.assertAlmostEqual(stats['val'], n_total * split_props[2],
         delta=n_total*0.05)

    def testClassAssignmentBalanced(self):
        id_label_dict1 = {str(x): {"primary": ["cat"]} for
                          x in range(0, int(20e3))}
        id_label_dict2 = {str(x): {"primary": ["dog"]} for
                          x in range(int(20e3), int(25e3))}
        id_label_dict3 = {str(x): {"primary": ["ele"]} for
                          x in range(int(25e3), int(27e3))}
        id_label_dict = {**id_label_dict1, **id_label_dict2, **id_label_dict3}

        split_names = ["train", "test", "val"]
        split_props = [0.5, 0.3, 0.2]
        labels_all = ["cat", "dog", "ele"]

        set_assignment = self.splitter._assign_id_to_set(
            id_label_dict,
            split_names=split_names,
            split_props=split_props,
            balanced_sampling_min=True,
            balanced_sampling_label_type="primary"
        )

        stats = {x: 0 for x in labels_all}
        for k, v in set_assignment.items():
            label = id_label_dict[k]['primary'][0]
            stats[label] += 1

        self.assertAlmostEquals(stats['cat'], stats['dog'],
            delta=5)
        self.assertAlmostEquals(stats['cat'], stats['ele'],
            delta=5)
        self.assertAlmostEquals(stats['ele'], len(id_label_dict3.keys()),
            delta=5)

    def testClassAssignmentBalancedMultiType(self):
        id_label_dict1 = {str(x): {"primary": ["cat"], "color": ["brown"]} for
                          x in range(0, int(20e3))}
        id_label_dict2 = {str(x): {"primary": ["dog"], "color": ["white"]} for
                          x in range(int(20e3), int(25e3))}
        id_label_dict3 = {str(x): {"primary": ["ele"], "color": ["green"]} for
                          x in range(int(25e3), int(27e3))}
        id_label_dict = {**id_label_dict1, **id_label_dict2, **id_label_dict3}

        split_names = ["train", "test", "val"]
        split_props = [0.5, 0.3, 0.2]
        labels_all = ["cat", "dog", "ele"]

        set_assignment = self.splitter._assign_id_to_set(
            id_label_dict,
            split_names=split_names,
            split_props=split_props,
            balanced_sampling_min=True,
            balanced_sampling_label_type="primary"
        )

        stats = {x: 0 for x in labels_all}
        for k, v in set_assignment.items():
            label = id_label_dict[k]['primary'][0]
            stats[label] += 1

        self.assertAlmostEquals(stats['cat'], stats['dog'],
            delta=5)
        self.assertAlmostEquals(stats['cat'], stats['ele'],
            delta=5)
        self.assertAlmostEquals(stats['ele'], len(id_label_dict3.keys()),
            delta=5)

    def testClassAssignmentBalancedMultiLabel(self):
        id_label_dict1 = {str(x): {"primary": ["cat"]} for
                          x in range(0, int(20e3))}
        id_label_dict2 = {str(x): {"primary": ["dog", "ele"]} for
                          x in range(int(20e3), int(25e3))}
        id_label_dict3 = {str(x): {"primary": ["ele"]} for
                          x in range(int(25e3), int(27e3))}
        id_label_dict = {**id_label_dict1, **id_label_dict2, **id_label_dict3}

        split_names = ["train", "test", "val"]
        split_props = [0.5, 0.3, 0.2]
        labels_all = ["cat", "dog", "ele"]

        set_assignment = self.splitter._assign_id_to_set(
            id_label_dict,
            split_names=split_names,
            split_props=split_props,
            balanced_sampling_min=True,
            balanced_sampling_label_type="primary"
        )

        stats = {x: 0 for x in labels_all}
        for k, v in set_assignment.items():
            for label in id_label_dict[k]['primary']:
                stats[label] += 1

        self.assertAlmostEquals(stats['cat'], stats['dog'],
            delta=5)
        self.assertGreaterEqual(stats['ele'], stats['cat'])


    def testRemoveLabelTypes(self):
        test_dict = {'1': {'labels/primary': ['cat', 'dog'], 'labels/color': ['white', 'gray']},
                     '2': {'labels/primary': ['elephant'], 'labels/color': ['black']},
                     '3': {'labels/primary': ['leopard'],
                           'labels/color': ['brown']}}
        tt = self.splitter._remove_label_types(test_dict, 'labels/primary')

        self.assertEqual(tt['3'],{'labels/color': ['brown']})
        self.assertEqual(tt['1'],{'labels/color': ['white', 'gray']})
        self.assertEqual(tt['2'],{'labels/color': ['black']})


    def testKeepOnlyLabels(self):
        test_dict = {'1': {'labels/primary': ['cat', 'dog']},
                     '2': {'labels/primary': ['elephant']},
                     '3': {'labels/primary': ['leopard', 'dog']}}

        tt = self.splitter._keep_only_labels(test_dict, {'labels/primary': ['dog']})

        self.assertEqual(tt['3'],{'labels/primary': ['dog']})
        self.assertEqual(tt['1'],{'labels/primary': ['dog']})
        self.assertNotIn('2', tt)

