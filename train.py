""" Main File for Training a Model

Allows for detailed configurations.

Example Usage:
---------------
python train.py \
-train_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-train_tfr_prefix train \
-val_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-val_tfr_prefix val \
-test_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-test_tfr_prefix test \
-class_mapping_json ./test_big/cats_vs_dogs/tfr_files/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs/run_outputs/ \
-model_save_dir ./test_big/cats_vs_dogs/model_save_dir/ \
-model cats_vs_dogs \
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 10 \
-starting_epoch 0
"""
import argparse
import logging
import os

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import (
    TensorBoard, EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau)

from config.config import ConfigLoader
from config.config_logging import setup_logging
from training.utils import copy_models_and_config_files
from training.prepare_model import create_model
from predicting.predictor import Predictor
from data.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data.reader import DatasetReader
from data.image import (
        preprocess_image)
from data.utils import (
        calc_n_batches_per_epoch, export_dict_to_json, read_json,
        n_records_in_tfr, find_files_with_ending,
        get_most_recent_file_from_files)


# Configure Logging
setup_logging()
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train_tfr_path", type=str, required=True,
        help="Path to directory that contains the training TFR files"
    )
    parser.add_argument(
        "-train_tfr_prefix", default="train", type=str,
        required=True,
        help="The prefix of the training TFR files (default train)")
    parser.add_argument(
        "-val_tfr_path", type=str, required=True,
        help="Path to directory that contains the validation TFR files")
    parser.add_argument(
        "-val_tfr_prefix", default="val", type=str,
        required=True,
        help="The prefix of the validation TFR files (default val)")
    parser.add_argument(
        "-test_tfr_path", type=str, required=False,
        help="Path to directory that contains the test TFR files (optional)")
    parser.add_argument(
        "-test_tfr_prefix", default="test", type=str,
        required=False,
        help="The prefix of the test TFR files (default test, optional)")
    parser.add_argument(
        "-class_mapping_json", type=str, required=True,
        help='Path to the json file containing the class mappings')
    parser.add_argument(
        "-run_outputs_dir", type=str, required=True,
        help="Path to a directory to store data during the training")
    parser.add_argument(
        "-model_save_dir", type=str, required=True,
        help='Path to a directory to store final model files')
    parser.add_argument(
        "-model", type=str, required=True,
        help="The model architecture to use for training\
             (see config/models.yaml)")
    parser.add_argument(
        "-labels", nargs='+', type=str, required=True,
        help='The labels to model')
    parser.add_argument(
        "-batch_size", type=int, default=128,
        help="The batch size for model training, if too large the model may\
              crash with an OOM error. Use values between 64 and 256")
    parser.add_argument(
        "-n_cpus", type=int, default=4,
        help="The number of cpus to use. Use all available if possible.")
    parser.add_argument(
        "-n_gpus", type=int, default=1,
        help='The number of GPUs to use (default 1)')
    parser.add_argument(
        "-buffer_size", type=int, default=32768,
        help='The buffer size to use for shuffling training records. Use \
              smaller values if memory is limited.')
    parser.add_argument(
        "-max_epochs", type=int, default=70,
        help="The max number of epochs to train the model")
    parser.add_argument(
        "-starting_epoch", type=int, default=0,
        help="The starting epoch number.")
    parser.add_argument(
        "-transfer_learning", default=False,
        action='store_true', required=False,
        help="Flag to specify that transfer learning should be used, with\
              frozen non-output layers")
    parser.add_argument(
        "-continue_training", default=False,
        action='store_true', required=False,
        help="Flag that training should be continued from a saved model.")
    parser.add_argument(
        "-fine_tuning", default=False,
        action='store_true', required=False,
        help="Flag to specify that transfer learning should be used, with\
              fine-tuning all layers")
    parser.add_argument(
        "-model_to_load", type=str, required=False, default="",
        help='Path to a model (.hdf5) when either continue_training,\
             transfer_learning or fine_tuning are specified, \
             if a directory is specified, \
             the most recent model in that directory is loaded')
    parser.add_argument(
        "-pre_processing", type=str, default="standard",
        required=False,
        help="Has currently no effect, standard image pre-processing \
             is applied")

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    ###########################################
    # Process Input ###########
    ###########################################

    # Load model config
    model_cfg = ConfigLoader('./config/models.yaml')

    assert args['model'] in model_cfg.cfg['models'], \
        "model %s not found in config/models.yaml" % args['model']

    image_processing = model_cfg.cfg['models'][args['model']]['image_processing']
    input_shape = (image_processing['output_height'],
                   image_processing['output_width'], 3)
    # Prepare labels to model
    output_labels = args['labels']
    output_labels_clean = ['label/' + x for x in output_labels]

    # Class to numeric mappings and number of classes per label
    class_mapping = read_json(args['class_mapping_json'])
    n_classes_per_label_dict = {c: len(class_mapping[o]) for o, c in
                                zip(output_labels, output_labels_clean)}
    n_classes_per_label = [n_classes_per_label_dict[x]
                           for x in output_labels_clean]

    # save class mapping file to current run puth
    export_dict_to_json(class_mapping,
                        args['run_outputs_dir'] + 'label_mappings.json')

    # TFR files
    def _find_tfr_files(path, prefix):
        """ Find all TFR files """
        files = os.listdir(path)
        tfr_files = [x for x in files if x.endswith('.tfrecord') and
                     prefix in x]
        tfr_paths = [os.path.join(*[path, x]) for x in tfr_files]
        return tfr_paths

    tfr_train = _find_tfr_files(args['train_tfr_path'],
                                args['train_tfr_prefix'])
    tfr_val = _find_tfr_files(args['val_tfr_path'], args['val_tfr_prefix'])

    if len(args['test_tfr_path']) > 0:
        TEST_SET = True
        tfr_test = _find_tfr_files(args['test_tfr_path'],
                                   args['test_tfr_prefix'])
        pred_output_json = args['run_outputs_dir'] + 'test_preds.json'
    else:
        TEST_SET = False

    # Create best model output name
    best_model_save_path = args['model_save_dir'] + 'best_model.hdf5'

    # Create prediction model output name
    pred_model_path = args['model_save_dir'] + 'prediction_model.hdf5'

    # Define path of model to load if only directory is specified
    if len(args['model_to_load']) > 0:
        if not args['model_to_load'].endswith('.hdf5'):
            if os.path.isdir(args['model_to_load']):
                model_files = find_files_with_ending(args['model_to_load'], '.hdf5')
                most_recent_model = get_most_recent_file_from_files(model_files)
                args['model_to_load'] = most_recent_model
                logging.debug("Loading most recent model file %s:"
                              % most_recent_model)

    ###########################################
    # CALC IMAGE STATS ###########
    ###########################################

    tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()

    logger.info("Create Dataset Reader")
    data_reader = DatasetReader(tfr_encoder_decoder.decode_record)

    # Calculate Dataset Image Means and Stdevs for a dummy batch
    logger.info("Get Dataset Reader for calculating datset stats")
    dataset = data_reader.get_iterator(
            tfr_files=tfr_train,
            batch_size=30,
            is_train=False,
            n_repeats=1,
            output_labels=output_labels,
            image_pre_processing_fun=preprocess_image,
            image_pre_processing_args={**image_processing,
                                       'is_training': False},
            buffer_size=args['buffer_size'],
            num_parallel_calls=args['n_cpus'])
    iterator = dataset.make_one_shot_iterator()
    batch_data = iterator.get_next()

    logger.info("Calculating image means and stdevs")
    with tf.Session() as sess:
        features, labels = sess.run(batch_data)

    # calculate and save image means and stdvs of each color channel
    # for pre processing purposes
    image_means = [round(float(x), 4) for x in
                   list(np.mean(features['images'],
                                axis=(0, 1, 2), dtype=np.float64))]
    image_stdevs = [round(float(x), 4) for x in
                    list(np.std(features['images'],
                                axis=(0, 1, 2), dtype=np.float64))]

    image_processing['image_means'] = image_means
    image_processing['image_stdevs'] = image_stdevs

    logger.info("Image Means: %s" % image_means)
    logger.info("Image Stdevs: %s" % image_stdevs)

    ###########################################
    # PREPARE DATA READER ###########
    ###########################################

    logger.info("Preparing Data Feeders")

    def input_feeder_train():
        return data_reader.get_iterator(
                    tfr_files=tfr_train,
                    batch_size=args['batch_size'],
                    is_train=True,
                    n_repeats=None,
                    output_labels=output_labels,
                    image_pre_processing_fun=preprocess_image,
                    image_pre_processing_args={
                        **image_processing,
                        'is_training': True},
                    buffer_size=args['buffer_size'],
                    num_parallel_calls=args['n_cpus'])

    def input_feeder_val():
        return data_reader.get_iterator(
                    tfr_files=tfr_val,
                    batch_size=args['batch_size'],
                    is_train=False,
                    n_repeats=None,
                    output_labels=output_labels,
                    image_pre_processing_fun=preprocess_image,
                    image_pre_processing_args={
                        **image_processing,
                        'is_training': False},
                    buffer_size=args['buffer_size'],
                    num_parallel_calls=args['n_cpus'])

    if TEST_SET:
        def input_feeder_test():
            return data_reader.get_iterator(
                        tfr_files=tfr_test,
                        batch_size=args['batch_size'],
                        is_train=False,
                        n_repeats=1,
                        output_labels=output_labels,
                        image_pre_processing_fun=preprocess_image,
                        image_pre_processing_args={
                            **image_processing,
                            'is_training': False},
                        buffer_size=args['buffer_size'],
                        num_parallel_calls=args['n_cpus'],
                        drop_batch_remainder=False)

    # Export Image Processing Settings
    export_dict_to_json({**image_processing,
                         'is_training': False},
                        args['run_outputs_dir'] + 'image_processing.json')

    logger.info("Calculating batches per epoch")
    n_records_train = n_records_in_tfr(tfr_train)
    n_batches_per_epoch_train = calc_n_batches_per_epoch(
        n_records_train, args['batch_size'])

    n_records_val = n_records_in_tfr(tfr_val)
    n_batches_per_epoch_val = calc_n_batches_per_epoch(
        n_records_val, args['batch_size'])

    if TEST_SET:
        n_records_test = n_records_in_tfr(tfr_test)
        n_batches_per_epoch_test = calc_n_batches_per_epoch(
            n_records_test, args['batch_size'], drop_remainder=False)

    ###########################################
    # CREATE MODEL ###########
    ###########################################

    logger.info("Preparing Model")

    model = create_model(
        model_name=args['model'],
        input_shape=input_shape,
        target_labels=output_labels_clean,
        n_classes_per_label_type=n_classes_per_label,
        n_gpus=args['n_gpus'],
        continue_training=args['continue_training'],
        transfer_learning=args['transfer_learning'],
        fine_tuning=args['fine_tuning'],
        path_of_model_to_load=args['model_to_load'])

    logger.debug("Final Model Architecture")
    for layer, i in zip(model.layers,
                        range(0, len(model.layers))):
        logger.debug("Layer %s: Name: %s Input: %s Output: %s" %
                     (i, layer.name, layer.input_shape,
                      layer.output_shape))

    logger.info("Preparing Callbacks and Monitors")

    ###########################################
    # MONITORS ###########
    ###########################################

    # stop model training if validation loss does not improve
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0, patience=5, verbose=0, mode='auto')

    # reduce learning rate if model progress plateaus
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=0,
        mode='auto',
        min_delta=0.0001, cooldown=2, min_lr=1e-5)

    # log validation statistics to a csv file
    csv_logger = CSVLogger(args['run_outputs_dir'] + 'training.log')

    # create model checkpoints after each epoch
    checkpointer = ModelCheckpoint(
        filepath=args['run_outputs_dir'] +
        'model_epoch_{epoch:02d}_loss_{val_loss:.2f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=False, mode='auto', period=1)

    # save best model
    checkpointer_best = ModelCheckpoint(
        filepath=args['run_outputs_dir'] + 'model_best.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=1)

    # write graph to disk
    tensorboard = TensorBoard(log_dir=args['run_outputs_dir'],
                              histogram_freq=0,
                              batch_size=args['batch_size'],
                              write_graph=True,
                              write_grads=False, write_images=False)

    callbacks_list = [early_stopping, reduce_lr_on_plateau, csv_logger,
                      checkpointer, checkpointer_best]

    ###########################################
    # MODEL TRAINING  ###########
    ###########################################

    logger.info("Start Model Training")

    history = model.fit(
        input_feeder_train(),
        epochs=args['max_epochs'],
        steps_per_epoch=n_batches_per_epoch_train,
        validation_data=input_feeder_val(),
        validation_steps=n_batches_per_epoch_val,
        callbacks=callbacks_list,
        initial_epoch=args['starting_epoch'])

    logger.info("Finished Model Training")

    ###########################################
    # SAVE BEST MODEL ###########
    ###########################################

    copy_models_and_config_files(
            model_source=args['run_outputs_dir'] + 'model_best.hdf5',
            model_target=best_model_save_path,
            files_path_source=args['run_outputs_dir'],
            files_path_target=args['model_save_dir'],
            copy_files=".json")

    ###########################################
    # PREDICT AND EXPORT TEST DATA ###########
    ###########################################

    if TEST_SET:
        logger.info("Starting to predict on test data")
        pred = Predictor(
                model_path=best_model_save_path,
                class_mapping_json=args['class_mapping_json'],
                pre_processing_json=args['run_outputs_dir'] +
                'image_processing.json',
                batch_size=args['batch_size'])

        pred.predict_from_dataset(
            dataset=input_feeder_test(),
            export_type='json',
            output_file=pred_output_json)

        logger.info("Finished predicting on test data, saved to: %s" %
                    pred_output_json)
