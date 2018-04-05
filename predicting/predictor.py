""" Predict Images using a trained Model """
import os
import json
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

from pre_processing.image_transformations import preprocess_image
from data_processing.utils import print_progress, export_dict_to_json


class Predictor(object):
    """ Class to Predict Images """

    def __init__(self, model_path, class_mapping_json, pre_processing_json,
                 batch_size=128):
        """ Args:
            model_path: full path to a trained Keras model
            class_mapping_json: full path to json class mapping file
            pre_processing_json: full path to pre processing json file
            batch_size: numer of images to process once at a time
        """
        self.model_path = model_path
        self.class_mapping_json = class_mapping_json
        self.pre_processing_json = pre_processing_json
        self.batch_size = batch_size
        self.class_mapping = None
        self.pre_processing = None

        # check file existence
        path_names = ['model_path', 'class_mapping_json',
                      'pre_processing_json']
        paths = [self.model_path, self.class_mapping_json,
                 self.pre_processing_json]
        for name, path in zip(path_names, paths):
            if not os.path.isfile(path):
                raise FileNotFoundError("File %s not found at %s" %
                                        (name, path))

        # read the config files
        with open(self.class_mapping_json, 'r') as json_file:
            self.class_mapping = json.load(json_file)

        with open(self.pre_processing_json, 'r') as json_file:
            self.pre_processing = json.load(json_file)

        # create numeric id to string class mapper
        self.id_to_class_mapping = dict()
        for label_type, label_mappings in self.class_mapping.items():
            id_to_class = {v: k for k, v in label_mappings.items()}
            self.id_to_class_mapping[label_type] = id_to_class

        print("Read following class mappings:")
        self._log_cfg(self.class_mapping)

        print("Read following pre processing options:")
        self._log_cfg(self.pre_processing)

        self.model = load_model(self.model_path)

    def _log_cfg(self, cfg):
        """ Print configuration file """
        for k, v in cfg.items():
            if isinstance(v, dict):
                print("  Reading values for entry: %s" % k)
                for kk, vv in v.items():
                    print("    Key: %s - Value: %s" % (kk, vv))
            else:
                print("  Key: %s - Value: %s" % (k, v))

    def _process_single_prediction(self, preds):
        """ Process a single prediction """

        result = dict()
        for label_type, class_mappings in self.id_to_class_mapping.items():

            class_preds = {class_mappings[id]: value for id, value in
                           enumerate(list(preds))}

            ordered_classes = sorted(class_preds, key=class_preds.get,
                                     reverse=True)

            top_label = ordered_classes[0]
            top_value = class_preds[top_label]

            result[label_type] = OrderedDict([
                'predicted_class': top_label,
                'prediction_value': top_value,
                'class_predictions': class_preds
                )
        return result

    def _predict_images(self, images_list):
        """ Calculate Predictions """

        n_total = len(images_list)
        n_processed = 0

        image_files_tf = tf.constant(images_list)
        batch = self._create_dataset_iterator(image_files_tf, self.batch_size)

        all_predictions = OrderedDict()

        with K.get_session() as sess:
            while True:
                try:
                    batch_data = sess.run(batch)
                    images = batch_data['images']
                    file_paths = batch_data['file_path']
                    preds = self.model.predict_on_batch(images)
                    for i in range(0, preds.shape[0]):
                        file = file_paths[i].decode('utf-8')
                        pred_single = \
                            self._process_single_prediction(preds[i, :])
                        all_predictions[file] = pred_single

                    n_processed += images.shape[0]
                    print_progress(n_processed, n_total)

                except tf.errors.OutOfRangeError:
                    print("")
                    print("Finished Predicting")
                    break

        return all_predictions

    def predict_image_dir(self, path_to_image_dir):
        """ Args:
            - path_to_image_dir: full path to directory containing images
                to predict
        """
        image_files = [path_to_image_dir + os.path.sep + x for
                       x in os.listdir(path_to_image_dir)]
        print("Found %s images in %s" %
              (len(image_files), path_to_image_dir))

        self.predictions = self._predict_images(image_files)

    def _create_dataset_iterator(self, image_paths, batch_size):
        """ Creates an iterator which iterates over the input images
            and applies the image transformations
        """
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: self._get_and_transform_image(
                                x, self.pre_processing))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        images, file_path = iterator.get_next()

        return {'images': images, 'file_path': file_path}

    def _get_and_transform_image(self, image_path, pre_proc_args):
        """ Returns a processed image """
        image_raw = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
        image_processed = preprocess_image(image_decoded, **pre_proc_args)
        return image_processed, image_path

    def export_predictions_json(self, file_path):
        """ Export Predictions to Json """
        assert self.pre_processing is not None, \
            "Predictions not available, predict first"

        print("Start writing file: %s" % file_path)

        export_dict_to_json(self.predictions, file_path)

        print("Finished writing file: %s" % file_path)

    def export_predictions_csv(self, file_path):
        """ Export predictions as CSV """

        print("Start writing file: %s" % file_path)
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            # Write Header
            header_row = ['file', 'predicted_class', 'prediction_value',
                          'class_predictions']
            csvwriter.writerow(header_row)

            for file_name, values in self.predictions.items():
                row_to_write = [file_name, values['predicted_class'],
                                values['prediction_value'],
                                values['class_predictions']]

                csvwriter.writerow(row_to_write)

        print("Finished writing file: %s" % file_path)
