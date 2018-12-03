import unittest
import tensorflow as tf
from camera_trap_classifier.data.utils import (
    hash_string,
    assign_hash_to_zero_one,
    calc_n_batches_per_epoch,
    clean_input_path,
    randomly_split_dataset,
    generate_synthetic_data,
    generate_synthetic_batch
)
import random
import os


class RandomSplitterTest(unittest.TestCase):
    """ Test Random Splitting of Datasets """

    def setUp(self):
        ids = [str(i) for i in range(0, 100)]
        split_names = ['train', 'test', 'val']
        split_percent = [0.5, 0.3, 0.2]
        self.splits = randomly_split_dataset(ids, split_names, split_percent)

        # Create label dictionary
        id_to_label = {k: 'blank' for k in ids}
        for i in range(10, 15):
            id_to_label[i] = 'species'

        self.splits_balanced = randomly_split_dataset(
            ids, split_names, split_percent, True, id_to_label)

    def checkSplitSizes(self):
        split_list = [v for v in self.splits.values()]

        n_train = len([x for x in split_list if x == 'train'])
        n_test = len([x for x in split_list if x == 'test'])
        n_val = len([x for x in split_list if x == 'val'])

        self.assertEqual(n_train, 50)
        self.assertEqual(n_test, 30)
        self.assertEqual(n_val, 20)

    def checkSplitSizesBalanced(self):
        split_list = [v for v in self.splits_balanced.values()]

        n_train = len([x for x in split_list if x == 'train'])
        n_test = len([x for x in split_list if x == 'test'])
        n_val = len([x for x in split_list if x == 'val'])

        self.assertEqual(n_train, 5)
        self.assertEqual(n_test, 3)
        self.assertEqual(n_val, 2)


class IdHasherTests(unittest.TestCase):
    """ Test Hash Function """
    def setUp(self):
        self.random_string = str(random.randint(-1e6, 1e6))
        self.random_int = random.randint(-1e6, 1e6)
        self.test_string = "ldlaldfd_dfdfldfdf_ssdfdf"
        self.test_empty = ""
        self.test_none = None
        self.rand_32string = str(random.randint(1e31, 1e32-1))

    def testIdenticalHash(self):
        self.assertEqual(hash_string(self.random_string),
                         hash_string(self.random_string))

    def testIntegerHash(self):
        self.assertEqual(hash_string(self.random_int),
                         hash_string(self.random_int))

    def testValueBetweenOneZero(self):
        t1 = assign_hash_to_zero_one(self.rand_32string)
        self.assertTrue(t1 >= 0 and t1 <= 1)

    def testValueType(self):
        t2 = assign_hash_to_zero_one(self.rand_32string)
        self.assertTrue(type(t2) is float)

    # def testInvalidInput(self):
    #     self.assertRaises(id_to_zero_one(self.test_empty))


class CalcTests(unittest.TestCase):
    """ Test Calculation Functions """

    def testNormalCase(self):
        n_total = 26
        batch_size = 5
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=False), 6)

    def testDropRemainderCaseNormal(self):
        n_total = 26
        batch_size = 5
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 5)

    def testEqual(self):
        n_total = 6600
        batch_size = 6600
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=False), 1)

    def testBatchLarger(self):
        n_total = 450
        batch_size = 660
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=False), 1)

    def testBoundary(self):
        n_total = 450
        batch_size = 451
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=False), 1)

        n_total = 450
        batch_size = 449
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=False), 2)

        n_total = 0
        batch_size = 10
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=False), 0)


class PathCleaningTests(unittest.TestCase):
    """ Test Path Cleaning """

    def setUp(self):
        self.os_sep = os.path.sep

        def pathBuilder(path_list):
            """ Build Path """
            path = [x + self.os_sep for x in path_list]
            path = self.os_sep + ''.join(path)
            return path

        self.normal_path = pathBuilder(["data", "images", "test"])

        self.no_path_sep_at_end = \
            pathBuilder(["data", "images", "test"])

        self.no_path_sep_at_end = \
            self.no_path_sep_at_end[0:-len(self.os_sep)]

    def testNormal(self):
        """ Check good case """
        self.assertEqual(self.normal_path,
                         clean_input_path(self.normal_path))

    def checkMissingSep(self):
        """ Check good case """
        self.assertEqual(self.normal_path,
                         clean_input_path(self.no_path_sep_at_end))


class GenerateSyntheticDataTests(tf.test.TestCase):
    """ Test Synthetic Data Generation """

    def setUp(self):
        self.image_shape = (224, 224, 3)
        self.labels = ['label/species', 'label/counts']
        self.n_classes = [10, 3]
        self.n_images = 1
        self.batch_size = 64

    def testSyntheticRecordGeneration(self):
        record = generate_synthetic_batch(
            batch_size=self.batch_size,
            image_shape=self.image_shape,
            labels=self.labels,
            n_classes=self.n_classes,
            n_images=self.n_images)

        self.assertIsInstance(record, tuple)
        self.assertIsInstance(record[0], dict)
        self.assertEqual(record[0]['images'].shape, (self.batch_size, ) + self.image_shape)
        for label in self.labels:
            self.assertEqual(record[1][label].shape, (self.batch_size, ))

    def testSyntheticDatasetGeneration(self):
        dataset = generate_synthetic_data(
            batch_size=self.batch_size,
            image_shape=self.image_shape,
            labels=self.labels,
            n_classes=self.n_classes,
            n_images=self.n_images)

        self.assertIsInstance(dataset, tf.data.Dataset)
        # self.assertEqual(dataset.output_shapes[0]['images'],
        # (None,) + self.image_shape)
