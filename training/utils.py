""" Utils for Model Training """
from config.config import logging
import numpy as np


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
                 stop_after_n_rounds, reduction_abs=None,
                 reduction_mult=None, min_lr=0, minimize=True):
        """ Reduce Lerarning Rate On Plateau
         Args:
            initial_lr (float): initial learning Rate
            reduce_after_n_rounds (int): number of rounds stagnant eval is
                allowed before learning is terminated
            stop_after_n_rounds (int): number of rounds after reduction without
                better results before stopping
            reduction_abs (float): absolute reduction in lr
                (either this or _mult)
            reduction_mult (float): factor to reduce lr (either this or _abs)
            min_lr: minimum learning rate
            minimize: whether to minimize the metric
        """
        self.initial_lr = initial_lr
        self.reduce_after_n_rounds = reduce_after_n_rounds
        self.stop_after_n_rounds = stop_after_n_rounds
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

        self._calc_learning_rate()

    def _reduce_lr(self):
        """ Reduce Learning Rate """
        old_lr = self.current_lr
        if self.reduction_abs is not None:
            self.current_lr = self.current_lr - self.reduction_abs
        else:
            self.current_lr = self.current_lr * self.reduction_mult

        self.current_lr = np.max([self.current_lr, self.min_lr])

        logging.info("Changing learning rate from %s to %s" %
                     (old_lr, self.current_lr))

    def _reset(self):
        """ Reset Internal Stats """
        self.current_lr = self.initial_lr

    def _calc_learning_rate(self):
        """ Calculate Learning Rate """

        n_patience_used = 0
        n_since_reduced = None
        result_history = list()

        self._reset()

        for i, res in enumerate(self.results):
            # return initial learning rate after first round
            result_history.append(res)

            if i == 0:
                current_min_res = res
                continue

            if res >= current_min_res:
                n_patience_used += 1

            if n_patience_used >= self.reduce_after_n_rounds:
                self._reduce_lr()
                n_since_reduced = 0
                n_patience_used = 0

            elif n_since_reduced is not None:
                if res >= current_min_res:
                    n_since_reduced += 1
                    if n_since_reduced >= self.stop_after_n_rounds:
                        self.stop_learning = True
                else:
                    n_since_reduced = 0

            if res < current_min_res:
                n_patience_used = 0

            current_min_res = np.min(result_history)
