import unittest
import os


from training.hooks import CSVLogger


class CSVLoggerTester(unittest.TestCase):
    """ Test Reducing Learning Rate On Plateau """

    def setUp(self):
        self.test_path = './test/test_files/test_csv_logger.csv'

    def tearDown(self):
        if os.path.isfile(self.test_path):
            os.remove(self.test_path)

    def testCreateNewFile(self):
        """ Create New CSV File """
        logger = CSVLogger(self.test_path, metrics_names=["a", "b", "c"])
        results = [3, 4.5567, 123]
        row_id = 0
        logger.addResults(row_id, results)

        with open(self.test_path) as csvfile:
            r1 = csvfile.readline()
            r2 = csvfile.readline()

        self.assertEqual(r1, 'epoch,a,b,c\n')
        self.assertEqual(r2, '0,3,4.5567,123\n')

    def testAddtoExistingFile(self):
        """ Create New CSV File """
        logger = CSVLogger(self.test_path, metrics_names=["a", "b", "c"])
        results = [3, 4.5567, 123]
        row_id = 0
        logger.addResults(row_id, results)
        logger.addResults(1, [1, 2, 3])

        with open(self.test_path) as csvfile:
            #csvreader = csv.reader(csvfile, delimiter=',')
            r1 = csvfile.readline()
            r2 = csvfile.readline()
            r3 = csvfile.readline()

        self.assertEqual(r3, '1,1,2,3\n')

    def testAddtoExistingFileNewLogger(self):
        """ Create New CSV File """
        logger = CSVLogger(self.test_path, metrics_names=["a", "b", "c"])
        results = [3, 4.5567, 123]
        row_id = 0
        logger.addResults(row_id, results)

        logger2 = CSVLogger(self.test_path, metrics_names=["a", "b", "c"])
        logger2.addResults(1, [1, 2, 3])

        with open(self.test_path) as csvfile:
            #csvreader = csv.reader(csvfile, delimiter=',')
            r1 = csvfile.readline()
            r2 = csvfile.readline()
            r3 = csvfile.readline()

        self.assertEqual(r3, '1,1,2,3\n')
