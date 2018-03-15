import unittest
from training.utils import ReduceLearningRateOnPlateau, EarlyStopping


class ReduceLeraningRateOnPlateauTester(unittest.TestCase):
    """ Test Reducing Learning Rate On Plateau """

    def testPatienceAndStopAfterValues(self):
        self.red = ReduceLearningRateOnPlateau(
            initial_lr=0.1,
            reduce_after_n_rounds=3,
            stop_after_n_rounds=2,
            reduction_mult=0.1,
            min_lr=0.00001,
            minimize=True
        )

        eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.39, 0.36,
                     0.37, 0.365, 0.365, 0.361, 0.360]

        for i, res in enumerate(eval_hist):
            self.red.addResult(res)
            if i == 7:
                self.assertAlmostEqual(self.red.current_lr, 0.01)
            if i < 7:
                self.assertAlmostEqual(self.red.current_lr, 0.1)
            if i >= 12:
                self.assertTrue(self.red.stop_learning)

    def testMinLearningRate(self):
        self.red = ReduceLearningRateOnPlateau(
            initial_lr=0.1,
            reduce_after_n_rounds=3,
            stop_after_n_rounds=2,
            reduction_mult=0.1,
            min_lr=0.00001,
            minimize=True
        )
        eval_hist = [x for x in range(0, 50)]

        for i, res in enumerate(eval_hist):
            self.red.addResult(res)
            self.assertGreaterEqual(self.red.current_lr, self.red.min_lr)


class EarlyStoppingTester(unittest.TestCase):
    """ Test Early stopping """

    def testEarlyStopping(self):
        self.es = EarlyStopping(
            stop_after_n_rounds=5,
            minimize=True
        )

        eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.41, 0.41, 0.41, 0.41, 0.41, 0.36,
                     0.37, 0.365, 0.365, 0.361, 0.360]

        for i, res in enumerate(eval_hist):
            self.es.addResult(res)
            if i >= 9:
                self.assertTrue(self.es.stop_training)
            if i < 9:
                self.assertFalse(self.es.stop_training)

    def testEarlyStoppingAlmost(self):
        self.es = EarlyStopping(
            stop_after_n_rounds=5,
            minimize=True
        )

        eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.36, #9
                     0.37, 0.365, 0.365, 0.361, 0.361, 0.35]

        for i, res in enumerate(eval_hist):
            self.es.addResult(res)
            if i >= 14:
                self.assertTrue(self.es.stop_training)
            if i < 14:
                self.assertFalse(self.es.stop_training)


#from training.utils import ReduceLearningRateOnPlateau, EarlyStopping
#
#red = EarlyStopping(
#            stop_after_n_rounds=5,
#            minimize=True
#        )
#eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.41, 0.41, 0.41, 0.41, 0.41, 0.36,
#                     0.37, 0.365, 0.365, 0.361, 0.360]
#for i, res in enumerate(eval_hist):
#    red.addResult(res)
#
