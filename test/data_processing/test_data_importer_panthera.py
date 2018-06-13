import unittest
from data_processing.data_importer import DatasetImporter


class ImportFromCSVPantheraTester(unittest.TestCase):
    """ Test Import from Json """

    def setUp(self):
        path = './test/test_files/panthera.csv'
        source_type = 'panthera_csv'
        params = {'path': path}
        self.importer = DatasetImporter().create(source_type, params)
        self.data = self.importer.import_from_source()

    def testNormalCase(self):
        self.assertEqual(self.data["record_bird_1"],
                         {'labels': [{'species': 'Bird',
                                      'count': '1'}],
                          'images': ["~/a/b/record_bird_1.JPG"]})

    def testNAinCounts(self):
        self.assertEqual(self.data["record_human_NA"],
                         {'labels': [{'species': 'HUMAN',
                                     'count': 'NA'}],
                          'images': ["~/a/b/record_human_NA.JPG"]})

    def testCountCats(self):
        self.assertEqual(self.data["record_elephant_10"]['labels'][0]['count'], '10')
        self.assertEqual(self.data["record_elephant_11"]['labels'][0]['count'], '11-50')
        self.assertEqual(self.data["record_elephant_50"]['labels'][0]['count'], '11-50')
        self.assertEqual(self.data["record_elephant_51"]['labels'][0]['count'], '51+')


    def testDuplicateSpeciesOnlyOnceInList(self):
        self.assertEqual(self.data["record_multi_LionLion"],
                         {'labels': [{'species': 'Lion',
                                     'count': '1'}],
                          'images': ["~/a/b/record_multi_LionLion.JPG"]})

    def testMultiSpeciesConsolidation(self):
        all_classes = [x['species'] for x in self.data["record_multi_ZebraGiraffe"]['labels']]
        self.assertIn('Zebra', all_classes)
        self.assertIn('Giraffe', all_classes)
        # self.assertIn(1, self.data["record_multi_ZebraGiraffe"]['labels']['counts'])
        # self.assertIn(2, self.data["record_multi_ZebraGiraffe"]['labels']['counts'])
        self.assertEqual(["~/a/b/record_multi_ZebraGiraffe.JPG"], self.data["record_multi_ZebraGiraffe"]['images'])


if __name__ == '__main__':

    unittest.main()
