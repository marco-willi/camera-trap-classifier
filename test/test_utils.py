import unittest
from data_processing.utils import (
    id_to_zero_one,
    hash_string,
    assign_hash_to_zero_one,
    calc_n_batches_per_epoch,
    clean_input_path
)
import random
import os


class IdHasherTests(unittest.TestCase):
    """ Test Hash Function """
    def setUp(self):
        print("Load CFG")
        self.random_string = str(random.randint(-1e6, 1e6))
        self.random_int = random.randint(-1e6, 1e6)
        self.test_string = "ldlaldfd_dfdfldfdf_ssdfdf"
        self.test_empty = ""
        self.test_none = None

    def testIdenticalHash(self):
        self.assertEqual(hash_string(self.random_string),
                         hash_string(self.random_string))

    def testIntegerHash(self):
        self.assertEqual(hash_string(self.random_int),
                         hash_string(self.random_int))

    def testValueBetweenOneZero(self):
        t1 = assign_hash_to_zero_one(self.random_string)
        self.assertTrue(t1 >= 0 and t1 <= 1)

    def testValueType(self):
        t2 = assign_hash_to_zero_one(self.random_string)
        self.assertTrue(type(t2) is float)

    # def testInvalidInput(self):
    #     self.assertRaises(id_to_zero_one(self.test_empty))


class CalcTests(unittest.TestCase):
    """ Test Calculation Functions """

    def testNormalCase(self):
        n_total = 26
        batch_size = 5
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 6)

    def testEqual(self):
        n_total = 6600
        batch_size = 6600
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 1)

    def testBatchLarger(self):
        n_total = 450
        batch_size = 660
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 1)

    def testBoundary(self):
        n_total = 450
        batch_size = 451
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 1)

        n_total = 450
        batch_size = 449
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 2)

        n_total = 0
        batch_size = 10
        self.assertEqual(calc_n_batches_per_epoch(n_total, batch_size), 0)


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
