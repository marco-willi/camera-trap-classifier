import unittest
from config.config import Config


class LoadConfigTestCase(unittest.TestCase):

    def setUp(self):
        print("Load CFG")
        self.cfg_obj = Config('test/test_files/test_config.yaml')
        self.cfg_obj.load_config()
        self.cfg = self.cfg_obj.get_config()

    def testTop(self):
        self.assertEqual(self.cfg['top_level_entry'], 'top_level_value')

    def testInteger(self):
        self.assertEqual(self.cfg['top_level_integer'], 333)

    def testTrue(self):
        self.assertEqual(self.cfg['top_level_true'], True)

    def testEmpty(self):
        self.assertEqual(self.cfg['top_level_empty'], None)

    def testFraction(self):
        self.assertEqual(self.cfg['top_level_fraction'], 0.4)

    def testSubLevels(self):
        self.assertEqual(self.cfg['top_level_1']['sub_level_1_1'],
                         'sub_level_value_1_1')

        self.assertEqual(self.cfg['top_level_1']['sub_level_1_2'],
                         'sub_level_value_1_2')

        self.assertEqual(self.cfg['top_level_1']['sub_level_1_3_empty'],
                         None)

        self.assertEqual(
            self.cfg['top_level_1']['sub_level_1_4']['sub_level_1_4_1'],
            'sub_level_1_4_1_value')

# if __name__ == '__main__':
#     unittest.main(argv=['first-arg-is-ignored'], exit=False)
