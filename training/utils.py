""" Utils for Model Training """
from config.config import logging
import numpy as np
import tensorflow as tf


class LearningRateSetter(tf.train.SessionRunHook):

    def __init__(self, lr):
        self.lr = lr

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
                        fetches=None,
                        feed_dict={'learning_rate:0': self.lr})


class EarlyStopping(object):
    def __init__(self, stop_after_n_rounds, minimize=True):
        """ Stop Training if Eval does not improve in
            n rounds
        """
        self.stop_after_n_rounds = stop_after_n_rounds
        self.results = list()
        self.stop_training = False
        self.minimize = minimize

        assert stop_after_n_rounds > 0, \
            "stop_after_n_rounds must be larger than 0"

    def addResult(self, result):
        """ Add a result """
        if self.minimize:
            self.results.append(result)
        else:
            self.results.append(result*-1)

        self._calc_stopping()

    def _calc_stopping(self):
        """ Calculate wheter to stop training or not """
        n_since_improvement = 0
        result_history = list()
        for i, res in enumerate(self.results):
            result_history.append(res)
            if i == 0:
                continue
            min_val = np.min(result_history)
            if res > min_val:
                n_since_improvement += 1
                if n_since_improvement >= self.stop_after_n_rounds:
                    self.stop_training = True
                    logging.info("Early Stopping Activated")
            else:
                n_since_improvement = 0


class ReduceLearningRateOnPlateau(object):
    def __init__(self, initial_lr, reduce_after_n_rounds,
                 patience_after_reduction, reduction_abs=None,
                 reduction_mult=None, min_lr=0, minimize=True):
        """ Reduce Lerarning Rate On Plateau
         Args:
            initial_lr (float): initial learning Rate
            reduce_after_n_rounds (int): number of rounds stagnant eval is
                allowed before learning is terminated
            patience_after_reduction (int): number of rounds after reduction
                to wait for better results before reducing again
            reduction_abs (float): absolute reduction in lr
                (either this or _mult)
            reduction_mult (float): factor to reduce lr (either this or _abs)
            min_lr: minimum learning rate
            minimize: whether to minimize the metric
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.reduce_after_n_rounds = reduce_after_n_rounds
        self.patience_after_reduction = patience_after_reduction
        self.reduction_abs = reduction_abs
        self.reduction_mult = reduction_mult
        self.min_lr = min_lr
        self.results = list()
        self.minimize = minimize
        self.stop_learning = False

        assert self.reduction_abs is None or \
            self.reduction_mult is None, \
            "Either reduction_abs or reduction_mult has to be None"

    def addResult(self, result):
        """ Add a result """
        if self.minimize:
            self.results.append(result)
        else:
            self.results.append(result*-1)

        old_lr = self.current_lr
        self._calc_learning_rate()
        if old_lr != self.current_lr:
            logging.info("Changing learning rate from %s to %s" %
                         (old_lr, self.current_lr))

    def _reduce_lr(self):
        """ Reduce Learning Rate """

        if self.reduction_abs is not None:
            self.current_lr = self.current_lr - self.reduction_abs
        else:
            self.current_lr = self.current_lr * self.reduction_mult

        self.current_lr = np.max([self.current_lr, self.min_lr])

    def _reset(self):
        """ Reset Internal Stats """
        self.current_lr = self.initial_lr

    def _calc_learning_rate(self):
        """ Calculate Learning Rate """

        n_patience_used = 0
        n_since_reduced = None
        block_reduction = False
        result_history = list()

        self._reset()

        for i, res in enumerate(self.results):
            # return initial learning rate after first round
            result_history.append(res)

            if i == 0:
                current_min_res = res
                continue

            if (n_since_reduced is not None) and \
               (n_since_reduced < self.patience_after_reduction):
                block_reduction = True
            else:
                block_reduction = False

            no_improvement = (res >= current_min_res)

            if no_improvement:
                n_patience_used += 1
            else:
                n_patience_used = 0
                n_since_reduced = None

            if (n_patience_used >= self.reduce_after_n_rounds):
                if not block_reduction:
                    self._reduce_lr()
                    n_since_reduced = 0
                else:
                    n_since_reduced += 1

            current_min_res = np.min(result_history)
