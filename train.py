""" Main File for Training a Model

Allows for detailed configurations.

Example Usage:
---------------
python train.py \
-train_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-train_tfr_pattern train \
-val_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-val_tfr_pattern val \
-test_tfr_path ./test_big/cats_vs_dogs/tfr_files \
-test_tfr_pattern test \
-class_mapping_json ./test_big/cats_vs_dogs/tfr_files/label_mapping.json \
-run_outputs_dir ./test_big/cats_vs_dogs/run_outputs/ \
-model_save_dir ./test_big/cats_vs_dogs/model_save_dir/ \
-model small_cnn \
-labels class \
-batch_size 128 \
-n_cpus 2 \
-n_gpus 1 \
-buffer_size 512 \
-max_epochs 10 \
-starting_epoch 0 \
-color_augmentation full_randomized \
-optimizer sgd
"""
import argparse
import logging
import os

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import (
    TensorBoard, EarlyStopping, CSVLogger,  ReduceLROnPlateau)

from config.config import ConfigLoader
from config.config_logging import setup_logging
from training.utils import copy_models_and_config_files
from training.hooks import ModelCheckpoint, TableInitializerCallback
from training.prepare_model import create_model
from predicting.predictor import Predictor
from data.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data.reader import DatasetReader
from data.image import preprocess_image
from data.utils import (
    calc_n_batches_per_epoch, export_dict_to_json, read_json,
    n_records_in_tfr, n_records_in_tfr_parallel, find_files_with_ending,
    get_most_recent_file_from_files, find_tfr_files_pattern_subdir)


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train_tfr_path", type=str, required=True,
        help="Path to directory that contains the training TFR files \
              (incl. subdirs)")
    parser.add_argument(
        "-train_tfr_pattern", nargs='+', type=str,
        help="The pattern of the training TFR files (default train) \
              list of 1 or more patterns that all have to match",
        default=['train'], required=False)
    parser.add_argument(
        "-val_tfr_path", type=str, required=True,
        help="Path to directory that contains the validation TFR files \
              (incl. subdirs)")
    parser.add_argument(
        "-val_tfr_pattern", nargs='+', type=str,
        help="The pattern of the validation TFR files (default val) \
              list of 1 or more patterns that all have to match",
        default=['val'], required=False)
    parser.add_argument(
        "-test_tfr_path", type=str, required=False,
        help="Path to directory that contains the test TFR files \
              (incl. subdirs - optional)")
    parser.add_argument(
        "-test_tfr_pattern", nargs='+', type=str,
        help="The pattern of the test TFR files (default test) \
              list of 1 or more patterns that all have to match",
        default=['test'], required=False)
    parser.add_argument(
        "-class_mapping_json", type=str, required=True,
        help='Path to the json file containing the class mappings')
    parser.add_argument(
        "-run_outputs_dir", type=str, required=True,
        help="Path to a directory to store data during the training")
    parser.add_argument(
        "-log_outdir", type=str, required=False, default=None,
        help="Directory to write logfiles to (defaults to run_outputs_dir)")
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
        "-labels_loss_weights", nargs='+', type=float, default=None,
        help='A list of length labels indicating weights for the different\
              labels applied during model training')
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
        help="The starting epoch number (0-based index).")
    # Model Training Parameters
    parser.add_argument(
        "-initial_learning_rate", type=float, default=0.01,
        help="The initial learning rate.")
    parser.add_argument(
        "-optimizer", type=str, default="sgd",
        required=False,
        help="Which optimizer to use in training the model (sgd or rmsprop)")
    # Transfer-Learning and Model Loading
    parser.add_argument(
        "-transfer_learning", default=False,
        action='store_true', required=False,
        help="Option to specify that transfer learning should be used.")
    parser.add_argument(
        "-transfer_learning_type", default='last_layer', required=False,
        help="Option to specify that transfer learning should be used, by\
              allowing to adapt only the last layer ('last_layer') \
              or all layers ('all_layers') - default is 'last_layer'")
    parser.add_argument(
        "-continue_training", default=False,
        action='store_true', required=False,
        help="Flag that training should be continued from a saved model.")
    parser.add_argument(
        "-rebuild_model", default=False,
        action='store_true', required=False,
        help="Flag that model should be rebuild (if continue_training). \
              This might be necessary if model training should be continued\
              with different options (e.g. no GPUs, or different optimizer)")
    parser.add_argument(
        "-fine_tuning", default=False,
        action='store_true', required=False,
        help="Flag to specify that transfer learning should be used, with\
              fine-tuning all layers")
    parser.add_argument(
        "-model_to_load", type=str, required=False, default=None,
        help='Path to a model (.hdf5) when either continue_training,\
             transfer_learning or fine_tuning are specified, \
             if a directory is specified, \
             the most recent model in that directory is loaded')
    # Image Processing
    parser.add_argument(
        "-color_augmentation", type=str, default="full_randomized",
        required=False,
        help="Which (random) color augmentation to perform during model\
              training - choose one of:\
              [None, 'little', 'full_fast', 'full_randomized']. \
              This can slow down the pre-processing speed and starve the \
              GPU of data. Use None or little/full_fast options if input \
              pipeline is slow. Else full_randomized is recommended.")
    parser.add_argument(
        "-ignore_aspect_ratio", default=False, action='store_true',
        help="Wheter to ignore the aspect ratio of the images during model \
              training. This can improve the total area of the image the \
              model sees during training and prediction. However, the images \
              are slightly distorted with this option since they are \
              converted to squares.")

    # Parse command line arguments
    args = vars(parser.parse_args())

    # Configure Logging
    if args['log_outdir'] is None:
        args['log_outdir'] = args['run_outputs_dir']

    setup_logging(log_output_path=args['log_outdir'])

    logger = logging.getLogger(__name__)

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s: %s" % (k, v))

    ###########################################
    # Process Input ###########
    ###########################################

    # Load model config
    model_cfg = ConfigLoader('./config/models.yaml')

    assert args['model'] in model_cfg.cfg['models'], \
        "model %s not found in config/models.yaml" % args['model']

    image_processing = model_cfg.cfg['models'][args['model']]['image_processing']
    image_processing['ignore_aspect_ratio'] = args['ignore_aspect_ratio']

    input_shape = (image_processing['output_height'],
                   image_processing['output_width'], 3)

    # Add 'label/' prefix to labels as they are stored in the .tfrecord files
    output_labels = args['labels']
    output_labels_clean = ['label/' + x for x in output_labels]

    # Class to numeric mappings and number of classes per label
    class_mapping = read_json(args['class_mapping_json'])
    n_classes_per_label_dict = {c: len(class_mapping[o]) for o, c in
                                zip(output_labels, output_labels_clean)}
    n_classes_per_label = [n_classes_per_label_dict[x]
                           for x in output_labels_clean]

    # save class mapping file to current run path
    export_dict_to_json(
        class_mapping,
        os.path.join(args['run_outputs_dir'], 'label_mappings.json'))

    # Find TFR files
    tfr_train = find_tfr_files_pattern_subdir(
        args['train_tfr_path'],
        args['train_tfr_pattern'])
    tfr_val = find_tfr_files_pattern_subdir(
        args['val_tfr_path'],
        args['val_tfr_pattern'])

    if len(args['test_tfr_path']) > 0:
        TEST_SET = True
        tfr_test = find_tfr_files_pattern_subdir(
            args['test_tfr_path'],
            args['test_tfr_pattern'])
        pred_output_json = os.path.join(args['run_outputs_dir'],
                                        'test_preds.json')
    else:
        TEST_SET = False

    # Create best model output name
    best_model_save_path = os.path.join(args['model_save_dir'],
                                        'best_model.hdf5')

    # Define path of model to load if only directory is specified
    if args['model_to_load'] is not None:
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
    n_records_train = n_records_in_tfr_parallel(tfr_train, args['n_cpus'])
    dataset = data_reader.get_iterator(
            tfr_files=tfr_train,
            batch_size=min([4096, n_records_train]),
            is_train=True,
            n_repeats=1,
            output_labels=output_labels,
            label_to_numeric_mapping=class_mapping,
            image_pre_processing_fun=preprocess_image,
            image_pre_processing_args={**image_processing,
                                       'is_training': False},
            buffer_size=args['buffer_size'],
            num_parallel_calls=args['n_cpus'])
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()

    logger.info("Calculating image means and stdevs")
    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(iterator.initializer)
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
                    label_to_numeric_mapping=class_mapping,
                    image_pre_processing_fun=preprocess_image,
                    image_pre_processing_args={
                        **image_processing,
                        'is_training': True,
                        'color_augmentation': args['color_augmentation']},
                    buffer_size=args['buffer_size'],
                    num_parallel_calls=args['n_cpus'])

    def input_feeder_val():
        return data_reader.get_iterator(
                    tfr_files=tfr_val,
                    batch_size=args['batch_size'],
                    is_train=False,
                    n_repeats=None,
                    output_labels=output_labels,
                    label_to_numeric_mapping=class_mapping,
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
                        os.path.join(args['run_outputs_dir'],
                                     'image_processing.json'))

    logger.info("Calculating batches per epoch")
    n_batches_per_epoch_train = calc_n_batches_per_epoch(
        n_records_train, args['batch_size'])

    n_records_val = n_records_in_tfr_parallel(tfr_val, args['n_cpus'])
    n_batches_per_epoch_val = calc_n_batches_per_epoch(
        n_records_val, args['batch_size'])

    if TEST_SET:
        n_records_test = n_records_in_tfr_parallel(tfr_test, args['n_cpus'])
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
        rebuild_model=args['rebuild_model'],
        transfer_learning=args['transfer_learning'],
        transfer_learning_type=args['transfer_learning_type'],
        path_of_model_to_load=args['model_to_load'],
        initial_learning_rate=args['initial_learning_rate'],
        output_loss_weights=args['labels_loss_weights'])

    logger.debug("Final Model Architecture")
    for layer, i in zip(model.layers,
                        range(0, len(model.layers))):
        logger.debug("Layer %s: Name: %s Input: %s Output: %s" %
                     (i, layer.name, layer.input_shape,
                      layer.output_shape))

    logger.info("Preparing Callbacks and Monitors")

    ###########################################
    # MONITORS / HOOKS ###########
    ###########################################

    # stop model training if validation loss does not improve
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0, patience=5, verbose=0, mode='auto')

    # reduce learning rate if model progress plateaus
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=4,
        verbose=0,
        mode='auto',
        min_delta=0.0001, cooldown=2, min_lr=1e-5)

    # log validation statistics to a csv file
    csv_logger = CSVLogger(args['run_outputs_dir'] + 'training.log',
                           append=args['continue_training'])

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

    # Initialize tables (lookup tables)
    table_init = TableInitializerCallback()

    callbacks_list = [early_stopping, reduce_lr_on_plateau, csv_logger,
                      checkpointer, checkpointer_best, table_init, tensorboard]

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
