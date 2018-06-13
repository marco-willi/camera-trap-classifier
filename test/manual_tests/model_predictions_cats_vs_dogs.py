""" Test some aspects of training cats vs dogs multi output model """


""" Main File for Training a Keras/Tensorflow Model"""
import logging
import os

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
# import matplotlib.pyplot as plt

from config.config import ConfigLoader
from config.config_logging import setup_logging
from training.utils import (
        ReduceLearningRateOnPlateau, EarlyStopping, CSVLogger,
        ModelCheckpointer, find_the_best_id_in_log, find_model_based_on_epoch,
        copy_models_and_config_files)
from training.model_library import create_model

from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
from pre_processing.image_transformations import (
        preprocess_image)
from data_processing.utils import (
        calc_n_batches_per_epoch, export_dict_to_json, read_json,
        n_records_in_tfr)

# Configure Logging
setup_logging()
logger = logging.getLogger(__name__)


args=dict()
args['train_tfr_path']  = './test_big/cats_vs_dogs_multi/tfr_files'
args['train_tfr_prefix'] = 'train'
args['val_tfr_path'] = './test_big/cats_vs_dogs_multi/tfr_files'
args['val_tfr_prefix'] = 'val'
args['test_tfr_path'] = './test_big/cats_vs_dogs_multi/tfr_files'
args['test_tfr_prefix'] = 'test'
args['class_mapping_json'] = './test_big/cats_vs_dogs_multi/tfr_files/label_mapping.json'
args['run_outputs_dir'] = './test_big/cats_vs_dogs_multi/run_outputs/'
args['model_save_dir'] = './test_big/cats_vs_dogs_multi/model_save_dir/'
args['model'] = 'cats_vs_dogs'
args['labels'] = ['species', 'standing']
args['batch_size'] = 12
args['n_cpus'] = 2
args['n_gpus'] = 1
args['buffer_size'] = 1
args['max_epochs'] = 70
args['starting_epoch'] = 0
args['continue_training'] = False
args['transfer_learning'] = False
args['model_to_load'] = None


args=dict()
args['train_tfr_path']  = './test_big/cats_vs_dogs/tfr_files'
args['train_tfr_prefix'] = 'train'
args['val_tfr_path'] = './test_big/cats_vs_dogs/tfr_files'
args['val_tfr_prefix'] = 'val'
args['test_tfr_path'] = './test_big/cats_vs_dogs/tfr_files'
args['test_tfr_prefix'] = 'test'
args['class_mapping_json'] = './test_big/cats_vs_dogs/tfr_files/label_mapping.json'
args['run_outputs_dir'] = './test_big/cats_vs_dogs/run_outputs/'
args['model_save_dir'] = './test_big/cats_vs_dogs/model_save_dir/'
args['model'] = 'cats_vs_dogs'
args['labels'] = ['class']
args['batch_size'] = 128
args['n_cpus'] = 2
args['n_gpus'] = 1
args['buffer_size'] = 1
args['max_epochs'] = 70
args['starting_epoch'] = 0
args['continue_training'] = False
args['transfer_learning'] = False
args['model_to_load'] = None




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

# Prepare labels to model
output_labels = args['labels']
output_labels_clean = ['label/' + x for x in output_labels]

# Class to numeric mappings and number of classes per label
class_mapping = read_json(args['class_mapping_json'])
n_classes_per_label_dict = {c: len(class_mapping[o]) for o, c in
                            zip(output_labels, output_labels_clean)}
n_classes_per_label = [n_classes_per_label_dict[x]
                       for x in output_labels_clean]

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
else:
    TEST_SET = False

# Create best model output name
best_model_path = args['model_save_dir'] + 'best_model.hdf5'

# Create prediction model output name
pred_model_path = args['model_save_dir'] + 'prediction_model.hdf5'

###########################################
# CALC IMAGE STATS ###########
###########################################

tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()

data_reader = DatasetReader(tfr_encoder_decoder.decode_record)

logger.info("Create Dataset Reader")
data_reader = DatasetReader(tfr_encoder_decoder.decode_record)


###########################################
# PREPARE DATA READER ###########
###########################################

logger.info("Preparing Data Feeders")

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
                    max_multi_label_number=None,
                    buffer_size=args['buffer_size'],
                    num_parallel_calls=args['n_cpus'],
                    return_only_ml_data=False)

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
        n_records_test, args['batch_size'])


############################
# TEST PREDICTOR
########################

# create numeric id to string class mapper
from collections import OrderedDict
import csv
import json
from data_processing.utils import (
    print_progress, export_dict_to_json, list_pictures,
    clean_input_path, get_file_name_from_path)

iterator = input_feeder_test()
model_path=pred_model_path
class_mapping_json=args['class_mapping_json']
pre_processing_json=args['run_outputs_dir'] + 'image_processing.json'
batch_size=args['batch_size']
output_csv = './test_big/cats_vs_dogs_multi/model_save_dir/test_preds.csv'

with open(pre_processing_json, 'r') as json_file:
    pre_processing = json.load(json_file)



############################
# FROM ITERATOR
########################

id_to_class_mapping = dict()
for label_type, label_mappings in class_mapping.items():
    id_to_class = {v: k for k, v in label_mappings.items()}
    id_to_class_mapping[label_type] = id_to_class


model = load_model(model_path)


all_predictions = OrderedDict()
output_names = model.output_names
id_to_class_mapping_clean = {'label/' + k: v for k, v in
                             id_to_class_mapping.items()}


with K.get_session() as sess:
    batch_data = sess.run(iterator)
    images = batch_data['images']
    ids = [x.decode('utf-8') for x in batch_data['id']]
    preds_list = model.predict_on_batch(images)
    if not isinstance(preds_list, list):
        preds_list = [preds_list]
    n_predictions = len(ids)

    for i, _id in enumerate(ids):
        id_preds = [x[i,:] for x in preds_list]
        result = dict()
        for o, output in enumerate(output_names):
            id_output_preds = id_preds[o]
            class_preds = {id_to_class_mapping_clean[output][ii]: y
                           for ii, y in enumerate(id_output_preds)}

            ordered_classes = sorted(class_preds, key=class_preds.get,
                                     reverse=True)

            top_label = ordered_classes[0]
            top_value = class_preds[top_label]
            result[output] = {
               'predicted_class': top_label,
               'prediction_value': top_value,
               'class_predictions': class_preds}

        all_predictions[_id] = result

    predictions = all_predictions


    with open(output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
            # Write Header
            header_row = ['id', 'label_type',
                          'predicted_class',
                          'prediction_value', 'class_predictions']
            csvwriter.writerow(header_row)



    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                               quoting=csv.QUOTE_ALL)
        for _id, values in predictions.items():
            for label_type, preds in values.items():
                row_to_write = [_id, label_type,
                                preds['predicted_class'],
                                preds['prediction_value'],
                                preds['class_predictions']]
                csvwriter.writerow(row_to_write)





#####################################
# Predict from image Paths
#####################################

import traceback
def _get_and_transform_image(image_path, pre_proc_args):
    """ Returns a processed image """
    image_raw = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_raw, channels=3,
                                         try_recover_truncated=True)
    image_processed = preprocess_image(image_decoded, **pre_proc_args)
    return image_processed, image_path


def _create_dataset_iterator(image_paths, batch_size):
    """ Creates an iterator interating over the input images
        and applying image transformations (resizing)
    """
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: _get_and_transform_image(
                          x, pre_processing))
    dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    images, file_path = iterator.get_next()

    return {'images': images, 'file_paths': file_path}



with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
    # Write Header
    header_row = ['id', 'label_type',
                  'predicted_class',
                  'prediction_value', 'class_predictions']
    csvwriter.writerow(header_row)



image_paths =  [x[0].decode('utf-8') for x in batch_data['image_paths']]

n_total = len(image_paths)
n_processed = 0

image_paths_tf = tf.constant(image_paths)
batch = _create_dataset_iterator(image_paths_tf, 3)


all_predictions = OrderedDict()

with K.get_session() as sess:
    while True:
        try:
            batch_data = sess.run(batch)
            batch_predictions = OrderedDict()
        except tf.errors.OutOfRangeError:
            print("")
            print("Finished Predicting")
            break
        except Exception as error:
            print("Failed to process batch with images:")
            max_processed = np.min([n_processed+3,
                                    n_total])
            for j in range(n_processed, max_processed):
                print("  Image in failed batch: %s" % image_paths[j])
            traceback.print_exc()
            continue


        images = batch_data['images']
        file_paths = batch_data['file_paths']

        preds_list = model.predict_on_batch(images)
        ids = [x.decode('utf-8') for x in batch_data['file_paths']]

        for i, _id in enumerate(ids):
            id_preds = [x[i,:] for x in preds_list]
            result = dict()
            for o, output in enumerate(output_names):
                id_output_preds = id_preds[o]
                class_preds = {id_to_class_mapping_clean[output][ii]: y
                               for ii, y in enumerate(id_output_preds)}

                ordered_classes = sorted(class_preds, key=class_preds.get,
                                         reverse=True)

                top_label = ordered_classes[0]
                top_value = class_preds[top_label]
                result[output] = {
                   'predicted_class': top_label,
                   'prediction_value': top_value,
                   'class_predictions': class_preds}

        all_predictions[_id] = result
        batch_predictions[_id] = result


        with open(output_csv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
            for _id, values in batch_predictions.items():
                for label_type, preds in values.items():
                    row_to_write = [_id, label_type,
                                    preds['predicted_class'],
                                    preds['prediction_value'],
                                    preds['class_predictions']]
                    csvwriter.writerow(row_to_write)


from predicting.predictor import Predictor
pred = Predictor(
        model_path=pred_model_path,
        class_mapping_json=args['class_mapping_json'],
        pre_processing_json=args['run_outputs_dir'] + 'image_processing.json',
        batch_size=args['batch_size'])

output_csv = './test_big/cats_vs_dogs_multi/model_save_dir/test_preds.csv'

pred.predict_from_iterator_and_export(input_feeder_test(), output_csv)


