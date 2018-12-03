""" Hooks / Callbacks that run during model training """
import csv
import os
import logging
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback

from camera_trap_classifier.training.utils import (
    is_multi_gpu_model, get_gpu_base_model)


class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        # assign model to save
        if is_multi_gpu_model(self.model):
            base_model = get_gpu_base_model(self.model)
            self.model_to_save = base_model
            self.model_to_save.optimizer = self.model.optimizer
            self.model_to_save.loss = self.model.loss
            self.model_to_save.metrics = self.model.metrics
            self.model_to_save.loss_weights = self.model.loss_weights
            self.model_to_save.sample_weight_mode = self.model.sample_weight_mode
            self.model_to_save.weighted_metrics = self.model.weighted_metrics
            self.model_to_save.target_tensors = self.model.target_tensors

        else:
            self.model_to_save = self.model

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)


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


class TableInitializerCallback(Callback):
    """ Initialize Tables - required with initializable tf.datasets
    """
    def on_train_begin(self, logs=None):
        K.get_session().run(tf.tables_initializer())


class ModelCheckpointer(Callback):
    """ Save model after each epoch """
    def __init__(self, model, path, save_model=True, save_weights=True):
        self.model_to_save = model
        self.path = path
        self.save_model = save_model
        self.save_weights = save_weights

    def on_epoch_end(self, epoch, logs=None):
        if self.save_model:
            save_id = 'model_epoch_%d.hdf5' % epoch
            save_path = os.path.join(self.path, save_id)
            self.model_to_save.save(save_path)

        if self.save_weights:
            save_id = 'model_weights_epoch_%d.hdf5' % epoch
            save_path = os.path.join(self.path, save_id)
            self.model_to_save.save_weights(save_path)


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
