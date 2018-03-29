import unittest
from data_processing.data_importer import ImportFromJson, ImportFromImageDirs, ImportFromPantheraCSV


class ImportFromCSVPantheraTester(unittest.TestCase):
    """ Test Import from Json """

    def setUp(self):
        self.csv_importer = ImportFromPantheraCSV()
        self.test_path_csv = './test/test_files/panthera.csv'
        self.data = self.csv_importer.read_from_csv(self.test_path_csv)

    def testNormalCase(self):
        self.assertEqual(self.data["record_bird_1"],
                         {'labels': {'species': ['Bird'],
                                     'counts': [1],
                                     'count_category': ['1']},
                          'images': ["~/a/b/record_bird_1.JPG"]})

    def testNAinCounts(self):
        self.assertEqual(self.data["record_human_NA"],
                         {'labels': {'species': ['HUMAN'],
                                     'counts': [-1],
                                     'count_category': ['NA']},
                          'images': ["~/a/b/record_human_NA.JPG"]})

    def testCountCats(self):
        self.assertEqual(self.data["record_elephant_10"]['labels']['count_category'], ['10'])
        self.assertEqual(self.data["record_elephant_11"]['labels']['count_category'], ['11-50'])
        self.assertEqual(self.data["record_elephant_50"]['labels']['count_category'], ['11-50'])
        self.assertEqual(self.data["record_elephant_51"]['labels']['count_category'], ['51+'])


    def testDuplicateSpeciesOnlyOnceInList(self):
        self.assertEqual(self.data["record_multi_LionLion"],
                         {'labels': {'species': ['Lion'],
                                     'counts': [1],
                                     'count_category': ['1']},
                          'images': ["~/a/b/record_multi_LionLion.JPG"]})

    def testMultiSpeciesConsolidation(self):
        self.assertIn('Zebra', self.data["record_multi_ZebraGiraffe"]['labels']['species'])
        self.assertIn('Giraffe', self.data["record_multi_ZebraGiraffe"]['labels']['species'])
        self.assertIn(1, self.data["record_multi_ZebraGiraffe"]['labels']['counts'])
        self.assertIn(2, self.data["record_multi_ZebraGiraffe"]['labels']['counts'])
        self.assertEqual(["~/a/b/record_multi_ZebraGiraffe.JPG"], self.data["record_multi_ZebraGiraffe"]['images'])


class ImportFromJsonTester(unittest.TestCase):
    """ Test Import from Json """

    def setUp(self):
        self.json_importer = ImportFromJson()
        self.test_path_json = './test/test_files/json_data_file.json'
        self.data = self.json_importer.read_from_json(self.test_path_json)

    def testLabelTypeIsNotListConvertedToList(self):
        self.assertIsInstance(
            self.data["no_list_label_type_primary"]["labels"]["primary"], list)

    def testEmptyImagesListNotInResults(self):
        self.assertNotIn("empty_images_list", self.data)

    def testEmptyImagesNotInResults(self):
        self.assertNotIn("only_empty_images_string", self.data)

    def testNonStringLabelNotInResult(self):
        self.assertNotIn("non_string_label", self.data)

    def testNormalCase(self):
        test_id = "10296725"
        self.assertIn(test_id, self.data)
        self.assertIn("labels", self.data[test_id])
        self.assertIn("images", self.data[test_id])
        self.assertEqual(self.data[test_id]['images'],
            ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
             "\\images\\4715\\all\\cat\\10296726_0.jpeg",
             "\\images\\4715\\all\\cat\\10296727_0.jpeg"
             ])

        self.assertEqual(
            self.data[test_id]['labels'],
            {"primary": ["cat"],
             "color": ["brown", "white", "black"]})


class ImportFromImageDirsTester(unittest.TestCase):
    """ Test Import from Image Directories """

    def setUp(self):
        self.image_dir_importer = ImportFromImageDirs()
        self.test_path_images = './test/test_images'
        self.data = self.image_dir_importer.read_from_image_root_dir(
            self.test_path_images)
        self.test_id_dog = 'dog3192'
        self.invalid_id = 'dog.3206'

    def testInstanceInDataSet(self):
        self.assertIn(self.test_id_dog, self.data)

    def testInvalidImageNameNotInResults(self):
        self.assertNotIn(self.invalid_id, self.data)

    def testInstanceHasCorrectClass(self):
        self.assertEqual(
            ['Dogs'],
            self.data[self.test_id_dog]['labels']['primary'])


if __name__ == '__main__':

    unittest.main()
