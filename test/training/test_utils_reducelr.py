import unittest
from training.hooks import ReduceLearningRateOnPlateau, EarlyStopping



#red = ReduceLearningRateOnPlateau3(
#            reduce_after_n_rounds=3,
#            patience_after_reduction=2,
#            reduction_mult=0.1,
#            min_lr=0.00001,
#            minimize=True)
#
#eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, #7, reduce
#             0.4, 0.39, # 9
#             0.36, 0.37, 0.365, 0.365, # 13 reduce here
#             0.361,
#             0.360,
#             0.361]
#
#mod_learn=0.1
#for i, res in enumerate(eval_hist):
#    #mod_learn = mod_learn*0.9
#    red.addResult(res, mod_learn)
#    if red.reduced_in_last_step:
#        mod_learn = red.new_lr
#    print("Iteration: %s, Mod LR: %s Reduced: %s" % (i, mod_learn, red.reduced_in_last_step))
#


class ReduceLeraningRateOnPlateauTester(unittest.TestCase):
    """ Test Reducing Learning Rate On Plateau """

    def testChangeLearningRateFlag(self):
        self.red = ReduceLearningRateOnPlateau(
            reduce_after_n_rounds=3,
            patience_after_reduction=2,
            reduction_mult=0.1,
            min_lr=0.00001,
            minimize=True
        )

        eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, #7, reduce
                     0.4, 0.39, # 9
                     0.36, 0.37, 0.365, 0.365, # 13 reduce here
                     0.361,
                     0.360,
                     0.361] # 16 Reduce again
        model_lr = 0.01

        for i, res in enumerate(eval_hist):
            self.red.addResult(res, model_lr)
            if i == 7:
                self.assertTrue(self.red.reduced_in_last_step)
            elif i == 13:
                self.assertTrue(self.red.reduced_in_last_step)
            elif i == 16:
                self.assertTrue(self.red.reduced_in_last_step)
            else:
                self.assertFalse(self.red.reduced_in_last_step)

    def testCorrectLearningRateCalculation(self):
        self.red = ReduceLearningRateOnPlateau(
            reduce_after_n_rounds=3,
            patience_after_reduction=2,
            reduction_mult=0.1,
            min_lr=0.00001,
            minimize=True
        )

        eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, #7, reduce
                     0.4, 0.39, # 9
                     0.36, 0.37, 0.365, 0.365, # 13 reduce here
                     0.361,
                     0.360,
                     0.361] # 16 Reduce again
        model_lr = 0.1

        for i, res in enumerate(eval_hist):
            self.red.addResult(res, model_lr)
            if self.red.reduced_in_last_step:
                model_lr = self.red.new_lr
            if i < 7:
                self.assertAlmostEqual(model_lr, 0.1)
            elif i < 13:
                self.assertAlmostEqual(model_lr, 0.01)
            elif i < 16:
                self.assertAlmostEqual(model_lr, 0.001)
            else:
                self.assertAlmostEqual(model_lr, 0.0001)

    def testMinLearningRate(self):
        self.red = ReduceLearningRateOnPlateau(
            reduce_after_n_rounds=3,
            patience_after_reduction=2,
            reduction_mult=0.1,
            min_lr=0.00001,
            minimize=True
        )

        eval_hist = [x for x in range(0, 50)]
        model_lr = 0.01

        for i, res in enumerate(eval_hist):
            self.red.addResult(res, model_lr)
            if self.red.reduced_in_last_step:
                self.assertGreaterEqual(self.red.new_lr, self.red.min_lr)




#class ReduceLeraningRateOnPlateauTester(unittest.TestCase):
#    """ Test Reducing Learning Rate On Plateau """
#
#    def testPatienceAndStopAfterValues(self):
#        self.red = ReduceLearningRateOnPlateau(
#            initial_lr=0.1,
#            reduce_after_n_rounds=3,
#            patience_after_reduction=2,
#            reduction_mult=0.1,
#            min_lr=0.00001,
#            minimize=True
#        )
#
#        eval_hist = [0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, #7, reduce
#                     0.4, 0.39, # 9
#                     0.36, 0.37, 0.365, 0.365, # 13 reduce here
#                     0.361,
#                     0.360,
#                     0.361] # 16 Reduce again
#
#        for i, res in enumerate(eval_hist):
#            self.red.addResult(res)
#            if i < 7:
#                self.assertAlmostEqual(self.red.current_lr, 0.1)
#            elif i < 13:
#                self.assertAlmostEqual(self.red.current_lr, 0.01)
#            elif i < 16:
#                self.assertAlmostEqual(self.red.current_lr, 0.001)
#            else:
#                self.assertAlmostEqual(self.red.current_lr, 0.0001)
#
#    def testMinLearningRate(self):
#        self.red = ReduceLearningRateOnPlateau(
#            initial_lr=0.1,
#            reduce_after_n_rounds=3,
#            patience_after_reduction=2,
#            reduction_mult=0.1,
#            min_lr=0.00001,
#            minimize=True
#        )
#        eval_hist = [x for x in range(0, 50)]
#
#        for i, res in enumerate(eval_hist):
#            self.red.addResult(res)
#            self.assertGreaterEqual(self.red.current_lr, self.red.min_lr)


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
