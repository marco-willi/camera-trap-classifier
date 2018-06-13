import unittest
from data_processing.data_importer import DatasetImporter


class ImportFromJsonTester(unittest.TestCase):
    """ Test Import from Json """

    def setUp(self):
        path = './test/test_files/json_data_file.json'
        source_type = 'json'
        params = {'path': path}
        self.importer = DatasetImporter().create(source_type, params)
        self.data = self.importer.import_from_source()

    def testSingleSpeciesStandard(self):
        self.assertEqual(
            self.data["single_species_standard"],
            {"labels": [{"class": "cat", "color_brown": "1", "color_white": "0", "counts": "1"}],
             "meta_data": {"meta_1": "meta_data_1",
                           "meta_2": "meta_data_2"},
             "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                        "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                        "\\images\\4715\\all\\cat\\10296727_0.jpeg"]
             })

    def testMultiSpeciesStandard(self):
        self.assertEqual(
            self.data["multi_species_standard"],
            {
           "labels": [
                      {"class": "cat", "color_brown": "1",
                       "color_white": "0", "counts": "1"},
                      {"class": "dog", "color_brown": "1",
                       "color_white": "0", "counts": "2"}
                      ],
            "meta_data": {"meta_1": "meta_data_1",
                          "meta_2": "meta_data_2"},
            "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296727_0.jpeg"
                      ]

          })

    def testEmptyClassImport(self):
        self.assertEqual(
            self.data["empty_image"],
            {
           "labels": [
                        {"class": "empty", "color_brown": "0",
                         "color_white": "0", "counts": "0"}
                      ],
            "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296727_0.jpeg"
                      ]
          }
        )


    def testMultiColorImport(self):
        self.assertEqual(
            self.data["single_species_multi_color"],
            {
           "labels": [
                        {"class": "cat", "color_brown": "1", "color_white": "1",
                         "counts": "1"}
                      ],
            "meta_data": {"meta_1": "meta_data_1",
                          "meta_2": "meta_data_2"},
            "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296727_0.jpeg"
                      ]

          }
        )

    def testEmptyImagesListNotInResults(self):
        self.assertNotIn("empty_images_list", self.data)

    def testMultiSpeciesNoOtherAttr(self):
        self.assertEqual(
            self.data["multi_species_no_other_attr"],
            {
           "labels": [
                        {"class": "cat"},
                        {"class": "dog"}
                      ],
            "meta_data": {"meta_1": "meta_data_1",
                          "meta_2": "meta_data_2"},
            "images": ["\\images\\4715\\all\\dog\\10300728_0.jpeg",
                       "\\images\\4715\\all\\dog\\10300742_0.jpeg"
                      ]

          }
        )

    def testNonStringLabelNotInResult(self):
        self.assertNotIn("non_string_label", self.data)

    def testInvalidMetaDataNotInResult(self):
        self.assertNotIn("meta_data_invalid", self.data)

    def testImportNoMetaData(self):
        self.assertEqual(
            self.data["no_meta_data"],
            {
            "labels": [
                        {"class": "cat", "color_brown": "1", "color_white": "0", "counts": "10-50"}
                       ],
            "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296727_0.jpeg"
                      ]
          }
        )

    def testEmptyImagesNotInResults(self):
        self.assertNotIn("only_empty_images_string", self.data)


if __name__ == '__main__':

    unittest.main()
