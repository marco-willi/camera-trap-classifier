import unittest
from data.importer import DatasetImporter


class ImportFromImageDirsTester(unittest.TestCase):
    """ Test Import from Image Directories """

    def setUp(self):
        path = './test/test_images'
        source_type = 'image_dir'
        params = {'path': path}
        self.importer = DatasetImporter().create(source_type, params)
        self.data = self.importer.import_from_source()
        self.test_id_dog = 'Dogs#dog3192'
        self.invalid_id = 'Dogs#dog.3206'

    def testInstanceInDataSet(self):
        self.assertIn(self.test_id_dog, self.data)

    def testInvalidImageNameNotInResults(self):
        self.assertNotIn(self.invalid_id, self.data)

    def testInstanceHasCorrectClass(self):
        self.assertEqual(
            'Dogs',
            self.data[self.test_id_dog]['labels'][0]['class'])


if __name__ == '__main__':

    unittest.main()
