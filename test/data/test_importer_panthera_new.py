import unittest
from data.importer import DatasetImporter


class ImportFromPantheraNew(unittest.TestCase):
    """ Test Import from new Panthera CSV """

    def setUp(self):
        path = './test/test_files/panthera_csv_new.csv'
        source_type = 'csv'
        params = {'path': path,
                  'image_path_col_list': 'dir',
                  'capture_id_col': 'capture_id',
                  'attributes_col_list': ['empty', 'species', 'count', 'standing', 'resting', 'moving', 'eating', 'interacting', 'babies', 'species_panthera'],
                  'meta_col_list': ['survey', 'image', 'location', 'split_name']}
        self.importer = DatasetImporter().create(
            source_type, params)
        self.data = self.importer.import_from_source()

    def testNormalCase(self):
        self.assertEqual(self.data['S1_20130521_20130704_1'],
                         {'labels': [{
                            'empty': 'species',
                            'species': 'otherbird',
                            'count': '1',
                            'standing': '-1',
                            'resting': '-1',
                            'moving': '-1',
                            'eating': '-1',
                            'interacting': '-1',
                            'babies': '-1',
                            'species_panthera': 'Bird'}],
                          'meta_data': {'survey': 'S1_20130521_20130704',
                            'image': 'S1__Station1__Camera1__CAM30817__2013-05-22__10-55-13.JPG',
                            'location': 'S1_20130521_20130704', 'split_name': 'test_S1_20130521_20130704'},
                          'images': ["/raw_images/S1_20130521_20130704/S1__Station1__Camera1__CAM30817__2013-05-22__10-55-13.JPG"]})

if __name__ == '__main__':

    unittest.main()
