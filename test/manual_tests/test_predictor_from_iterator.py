""" Test Predictor from Iterator"""
import os
import json

from pre_processing.image_transformations import preprocess_image
from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
from predicting.predictor import Predictor

model_path = './test_big/cats_vs_dogs_multi/model_save_dir/best_model.hdf5'
pre_processing_json = './test_big/cats_vs_dogs_multi/model_save_dir/image_processing.json'
class_mapping_json = './test_big/cats_vs_dogs_multi/model_save_dir/label_mappings.json'
image_dir = './test/test_images/'
tfr_path = './test_big/cats_vs_dogs_multi/tfr_files'
pred_output_csv = './test_big/cats_vs_dogs_multi/model_save_dir/preds.csv'
pred_output_json = './test_big/cats_vs_dogs_multi/model_save_dir/preds.json'
batch_size = 5


def _find_tfr_files(path, prefix):
    """ Find all TFR files """
    files = os.listdir(path)
    tfr_files = [x for x in files if x.endswith('.tfrecord') and
                 prefix in x]
    tfr_paths = [os.path.join(*[path, x]) for x in tfr_files]
    return tfr_paths


tfr_test = _find_tfr_files(tfr_path, 'test')
tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()
data_reader = DatasetReader(tfr_encoder_decoder.decode_record)

# read the config files
with open(class_mapping_json, 'r') as json_file:
    class_mapping = json.load(json_file)

with open(pre_processing_json, 'r') as json_file:
    pre_processing = json.load(json_file)

# create numeric id to string class mapper
id_to_class_mapping = dict()
for label_type, label_mappings in class_mapping.items():
    id_to_class = {v: k for k, v in label_mappings.items()}
    id_to_class_mapping[label_type] = id_to_class


def input_feeder_test():
    return data_reader.get_iterator(
                tfr_files=tfr_test,
                batch_size=12,
                is_train=False,
                n_repeats=1,
                output_labels=list(id_to_class_mapping.keys()),
                image_pre_processing_fun=preprocess_image,
                image_pre_processing_args={
                    **pre_processing,
                    'is_training': False},
                buffer_size=1,
                num_parallel_calls=2)


pred = Predictor(
            model_path=model_path,
            class_mapping_json=class_mapping_json,
            pre_processing_json=pre_processing_json,
            batch_size=batch_size)

pred.predict_from_dataset(
    dataset=input_feeder_test(),
    export_type='csv',
    output_file=pred_output_csv)

# pred.predict_from_dataset(
#     dataset=input_feeder_test(),
#     export_type='json',
#     output_file=pred_output_json)
