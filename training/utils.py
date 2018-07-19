""" Utils for Model Training """
import csv
import os
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback

from data_processing.utils import copy_file


class CSVLogger(object):
    """ Log stats to a csv """
    def __init__(self, path_to_logfile, metrics_names, row_id_name="epoch"):
        self.path_to_logfile = path_to_logfile
        self.metrics_names = metrics_names
        self.row_id_name = row_id_name

        assert isinstance(metrics_names, list), "metrics_names must be a list"

    def addResults(self, row_id, metrics):
        """ Add Metrics to Log File """
        row_to_write = [row_id] + metrics

        assert isinstance(metrics, list), "metrics must be a list"

        # Append if file exists
        if os.path.exists(self.path_to_logfile):
            with open(self.path_to_logfile, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(row_to_write)

        # Create new file if it does not exist
        else:
            with open(self.path_to_logfile, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                # Write Header
                header_row = [self.row_id_name] + self.metrics_names
                csvwriter.writerow(header_row)
                csvwriter.writerow(row_to_write)


class LearningRateSetter(tf.train.SessionRunHook):
    """ Hook to change learning rate in a TF Graph """
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


class ModelCheckpointer(Callback):
    """ Save model after each epoch """
    def __init__(self, model, path, save_model=True, save_weights=True):
        self.model_to_save = model
        self.path = path
        self.save_model = save_model
        self.save_weights = save_weights

    def on_epoch_end(self, epoch, logs=None):
        if self.save_model:
            self.model_to_save.save('%smodel_epoch_%d.hdf5' %
                                    (self.path, epoch))
        if self.save_weights:
            self.model_to_save.save_weights('%smodel_weights_epoch_%d.hdf5' %
                                            (self.path, epoch))


def find_the_best_id_in_log(log_file_path, metric, id='epoch', offset=-1):
    """ Returns the path of the best model """
    if not os.path.exists(log_file_path):
        raise FileExistsError("File %s does not exist" % log_file_path)

    epoch_results = dict()
    with open(log_file_path, newline='') as logfile:
        reader = csv.reader(logfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                metric_col = row.index(metric)
                id_col = row.index(id)
            else:
                epoch_results[row[id_col]] = float(row[metric_col])

    best_model_id = min(epoch_results, key=epoch_results.get)
    return best_model_id


def find_model_based_on_epoch(model_path, epoch, offset=-1):
    """ Returns path to the best model """
    epoch = str(int(epoch) + offset)
    files = [file_dir for file_dir in os.listdir(model_path)
             if os.path.isfile(os.path.join(model_path, file_dir))]
    all_model_files = [x for x in files if x.endswith('.hdf5')]
    search_string = 'epoch_' + epoch
    models_to_find = [x for x in all_model_files if search_string in x]
    path_to_models_to_find = [model_path + x for x in models_to_find]
    return path_to_models_to_find


def copy_models_and_config_files(model_source, model_target,
                                 files_path_source,
                                 files_path_target, copy_files=".json"):
    """ copy model from source to target and all config files """

    files = [file_dir for file_dir in os.listdir(files_path_source)
             if os.path.isfile(os.path.join(files_path_source, file_dir))]
    files_to_copy = [x for x in files if x.endswith(copy_files)]

    for file in files_to_copy:
        file_path = os.path.join(files_path_source, file)
        target_path = os.path.join(files_path_target, file)
        copy_file(file_path, target_path)

    copy_file(model_source, model_target)


class ReduceLearningRateOnPlateau(object):
    def __init__(self, reduce_after_n_rounds,
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
        self.reduce_after_n_rounds = reduce_after_n_rounds
        self.patience_after_reduction = patience_after_reduction
        self.reduction_abs = reduction_abs
        self.reduction_mult = reduction_mult
        self.min_lr = min_lr
        self.results = list()
        self.minimize = minimize
        self.stop_learning = False
        self.reduced_in_last_step = False
        self.new_lr = None

        assert self.reduction_abs is None or \
            self.reduction_mult is None, \
            "Either reduction_abs or reduction_mult has to be None"

    def addResult(self, result, current_model_lr):
        """ Add a result """

        self.reduced_in_last_step = False

        if self.minimize:
            self.results.append(result)
        else:
            self.results.append(result*-1)

        reduce = self._calc_if_reduction_needed()
        if reduce:
            new_lr = self._reduce_lr(current_model_lr)
            self.new_lr = new_lr

    def _reduce_lr(self, old_lr):
        """ Reduce Learning Rate """

        if self.reduction_abs is not None:
            new_lr = old_lr - self.reduction_abs
        else:
            new_lr = old_lr * self.reduction_mult

        new_lr = np.max([new_lr, self.min_lr])

        if not old_lr == new_lr:
            logging.info("Changing learning rate from %s to %s"
                         % (old_lr, new_lr))
            self.reduced_in_last_step = True
        return new_lr

    def _calc_if_reduction_needed(self):
        """ Calculate wheter lr has to be reduced """

        n_patience_used = 0
        n_since_reduced = None
        block_reduction = False
        result_history = list()

        for i, res in enumerate(self.results):
            # return initial learning rate after first round
            result_history.append(res)
            change_lr = False

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
                    change_lr = True
                    n_since_reduced = 0
                else:
                    n_since_reduced += 1

            current_min_res = np.min(result_history)
        return change_lr
