import unittest
from data_processing.data_importer import DatasetImporter


# test_path_csv = './test/test_files/panthera.csv'
# importer = DatasetImporter().create('panthera_csv', test_path_csv)
# data = importer.import_from_source()


class ImportFromCSVPantheraTester(unittest.TestCase):
    """ Test Import from Json """

    def setUp(self):
        path = './test/test_files/panthera.csv'
        type = 'panthera_csv'
        self.importer = DatasetImporter()
        self.source_importer = self.importer.create(type, path)
        self.data = self.source_importer.import_from_source()

    def testNormalCase(self):
        self.assertEqual(self.data["record_bird_1"],
                         {'labels': {'species': ['Bird'],
                                     'count_category': ['1']},
                          'images': ["~/a/b/record_bird_1.JPG"]})

    def testNAinCounts(self):
        self.assertEqual(self.data["record_human_NA"],
                         {'labels': {'species': ['HUMAN'],
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
                                     'count_category': ['1']},
                          'images': ["~/a/b/record_multi_LionLion.JPG"]})

    def testMultiSpeciesConsolidation(self):
        self.assertIn('Zebra', self.data["record_multi_ZebraGiraffe"]['labels']['species'])
        self.assertIn('Giraffe', self.data["record_multi_ZebraGiraffe"]['labels']['species'])
        # self.assertIn(1, self.data["record_multi_ZebraGiraffe"]['labels']['counts'])
        # self.assertIn(2, self.data["record_multi_ZebraGiraffe"]['labels']['counts'])
        self.assertEqual(["~/a/b/record_multi_ZebraGiraffe.JPG"], self.data["record_multi_ZebraGiraffe"]['images'])


class ImportFromJsonTester(unittest.TestCase):
    """ Test Import from Json """

    def setUp(self):
        path = './test/test_files/json_data_file.json'
        type = 'json'
        self.importer = DatasetImporter()
        self.source_importer = self.importer.create(type, path)
        self.data = self.source_importer.import_from_source()

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
        path = './test/test_images'
        type = 'image_dir'
        self.importer = DatasetImporter()
        self.source_importer = self.importer.create(type, path)
        self.data = self.source_importer.import_from_source()
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
