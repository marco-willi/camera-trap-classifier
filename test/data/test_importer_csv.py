import unittest
from data.importer import DatasetImporter


class ImportFromCSVSingleImageTester(unittest.TestCase):
    """ Test Import from CSV """

    def setUp(self):
        path = './test/test_files/dataset_single_image_multi_species.csv'
        source_type = 'csv'
        params = {'path': path,
                  'image_path_col_list': 'image',
                  'capture_id_col': 'capture_id',
                  'attributes_col_list': ['species', 'count', 'standing']}
        self.importer = DatasetImporter().create(
            source_type, params)
        self.data = self.importer.import_from_source()

    def testNormalCase(self):
        self.assertEqual(self.data["ele_1_0"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '1',
                                      'standing': '0'}],
                          'images': ["/path/capture_ele.jpg"]})

        self.assertEqual(self.data["zebra_2_1"],
                         {'labels': [{'species': 'Zebra',
                                      'count': '2',
                                      'standing': '1'}],
                          'images': ["/path/capture_zebra.jpg"]})

    def testCountCategoryImporters(self):

        self.assertEqual(self.data["wild_1050_0"],
                         {'labels': [{'species': 'Wildebeest',
                                      'count': '10-50',
                                      'standing': '0'}],
                          'images': ["/path/capture_wilde.jpg"]})

        self.assertEqual(self.data["wild_50+_1"],
                         {'labels': [{'species': 'Wildebeest',
                                      'count': '50+',
                                      'standing': '1'}],
                          'images': ["/path/capture_wilde2.jpg"]})

    def testMultiSpeciesCase(self):

        self.assertEqual(self.data["ele_lion"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '1',
                                      'standing': '0'},
                                     {'species': 'Lion',
                                      'count': '2',
                                      'standing': '1'}],
                          'images': ["/path/capture_ele_lion.jpg"]})

    def testMissingFields(self):
        """ Test Removal of Records with missing fields """
        self.assertNotIn('no_image', self.data)

    def testConvertMissingLabels(self):
        """ Test Removal of Records with missing fields """
        self.assertEqual(self.data['no_species']['labels'][0]['species'], '-1')
        self.assertEqual(self.data['no_count']['labels'][0]['count'], '-1')
        self.assertEqual(self.data['no_standing']['labels'][0]['standing'], '-1')

    def testInconsistendImage(self):
        self.assertEqual(self.data["ele_zebra_diff_image"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '2',
                                      'standing': '0'},
                                     {'species': 'Zebra',
                                      'count': '3',
                                      'standing': '0'}],
                          'images': ["/path/capture_ele_zebra.jpg"]})


class ImportFromCSVMultiImageTester(unittest.TestCase):
    """ Test Import from CSV """

    def setUp(self):
        path = './test/test_files/dataset_multi_image_multi_species.csv'
        source_type = 'csv'
        params = {'path': path,
                  'image_path_col_list': ['image1', 'image2', 'image3'],
                  'capture_id_col': 'capture_id',
                  'attributes_col_list': ['species', 'count', 'standing']}
        self.importer = DatasetImporter().create(
            source_type, params)
        self.data = self.importer.import_from_source()

    def testNormalCase(self):
        self.assertEqual(self.data["ele_1_0"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '1',
                                      'standing': '0'}],
                          'images': ["/path/capture_ele1.jpg",
                                     "/path/capture_ele2.jpg",
                                     "/path/capture_ele3.jpg"]})

    def testMultiSpeciesCase(self):

        self.assertEqual(self.data["ele_lion"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '1',
                                      'standing': '0'},
                                     {'species': 'Lion',
                                      'count': '2',
                                      'standing': '1'}],
                          'images': ["/path/capture_ele_lion1.jpg",
                                     "/path/capture_ele_lion2.jpg",
                                     "/path/capture_ele_lion3.jpg"]})

    def testNotAllImages(self):
        self.assertEqual(self.data["only_one_image"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '1',
                                      'standing': '0'}],
                          'images': ["/path/capture_ele3.jpg"]})

        self.assertEqual(self.data["only_two_images"],
                         {'labels': [{'species': 'Elephant',
                                      'count': '1',
                                      'standing': '0'}],
                          'images': ["/path/capture_ele1.jpg",
                                     "/path/capture_ele2.jpg"]})

if __name__ == '__main__':

    unittest.main()
