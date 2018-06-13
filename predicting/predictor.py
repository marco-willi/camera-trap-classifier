""" Predict Images using a trained Model """
import os
import json
import csv
from collections import OrderedDict
import traceback

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

from pre_processing.image_transformations import preprocess_image
from data_processing.utils import (
    print_progress, export_dict_to_json, list_pictures,
    clean_input_path, get_file_name_from_path)


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

            result[label_type] = {
                'predicted_class': top_label,
                'predicted_value': top_value,
                'class_predictions': class_preds}
        return result

    def _predict_images_and_export_preds(self, image_paths, export_path):
        """ Calculate Predictions Batch by Batch"""

        n_total = len(image_paths)
        n_processed = 0

        output_names = self.model.output_names
        id_to_class_mapping_clean = {'label/' + k: v for k, v in
                                     self.id_to_class_mapping.items()}
        image_paths_tf = tf.constant(image_paths)
        batch = self._create_dataset_iterator(image_paths_tf, self.batch_size)

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
                    max_processed = np.min([n_processed+self.batch_size,
                                            n_total])
                    for j in range(n_processed, max_processed):
                        print("  Image in failed batch: %s" % image_paths[j])
                    traceback.print_exc()
                    continue
                images = batch_data['images']
                preds_list = self.model.predict_on_batch(images)
                if not isinstance(preds_list, list):
                    preds_list = [preds_list]
                ids = [x.decode('utf-8') for x in batch_data['file_paths']]

                for i, _id in enumerate(ids):
                    id_preds = [x[i, :] for x in preds_list]
                    result = dict()
                    for o, output in enumerate(output_names):
                        id_output_preds = id_preds[o]
                        class_preds = {id_to_class_mapping_clean[output][ii]: y
                                       for ii, y in enumerate(id_output_preds)}

                        ordered_classes = sorted(class_preds,
                                                 key=class_preds.get,
                                                 reverse=True)

                        top_label = ordered_classes[0]
                        top_value = class_preds[top_label]
                        result[output] = {
                           'predicted_class': top_label,
                           'predicted_value': top_value,
                           'class_predictions': class_preds}

                    all_predictions[_id] = result
                    batch_predictions[_id] = result

                # append batch predictions to export here
                self._append_predictions_to_csv(batch_predictions, export_path)

                n_processed += images.shape[0]
                print_progress(n_processed, n_total)

        self.predictions = all_predictions

    def predict_image_dir_and_export(self, path_to_image_dir,
                                     export_file, check_images_first=0):
        """ Args:
        - path_to_image_dir: full path to directory containing images and
            subdirectories with images to predict
        - export_file: path to write export file to
        - check_images_first: whether to check each image for corruption
            before starting to predict (this is usually not necessary)
        """

        path_to_image_dir = clean_input_path(path_to_image_dir)
        image_paths = list_pictures(path_to_image_dir,  ext='jpg|jpeg')

        print("Found %s images in %s" %
              (len(image_paths), path_to_image_dir))

        if check_images_first == 1:
            print("Starting to check all images")
            image_paths = self._check_all_images(image_paths)

        self._create_csv_file_with_header(export_file)
        self._predict_images_and_export_preds(image_paths, export_file)

    def predict_from_iterator_and_export(self, iterator, export_file):
        """ Args:
        - iterator: iterator object
        - export_file: path to write export file to
        """
        self._create_csv_file_with_header(export_file)
        self._predict_from_iterator_and_export_to_csv(iterator, export_file)

    def _create_dataset_iterator(self, image_paths, batch_size):
        """ Creates an iterator interating over the input images
            and applying image transformations (resizing)
        """
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: self._get_and_transform_image(
                              x, self.pre_processing))
        dataset = dataset.apply(tf.contrib.data.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        images, file_path = iterator.get_next()

        return {'images': images, 'file_paths': file_path}

    def _check_all_images(self, image_list):
        """ Check imags for corruption """
        good_images = list()
        n_total = len(image_list)
        for counter, image in enumerate(image_list):
            print_progress(counter, n_total)
            try:
                img = Image.open(image)
                img.thumbnail([self.pre_processing['output_height'],
                               self.pre_processing['output_width']],
                              Image.ANTIALIAS)
                img.close()
                good_images.append(image)
            except (IOError, SyntaxError) as e:
                print('corrupt image - skipping:', image)
        return good_images

    def _get_and_transform_image(self, image_path, pre_proc_args):
        """ Returns a processed image """
        image_raw = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_raw, channels=3,
                                             try_recover_truncated=True)
        image_processed = preprocess_image(image_decoded, **pre_proc_args)
        return image_processed, image_path

    def _export_predictions_json(self, file_path):
        """ Export Predictions to Json """
        assert self.pre_processing is not None, \
            "Predictions not available, predict first"

        print("Start writing file: %s" % file_path)

        export_dict_to_json(self.predictions, file_path)

        print("Finished writing file: %s" % file_path)

    def _create_csv_file_with_header(self, file_path):
        """ Creates a csv file with a header row """
        print("Creating file: %s" % file_path)

        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
            # Write Header
            header_row = ['id', 'label',
                          'predicted_class',
                          'prediction_value', 'class_predictions']
            csvwriter.writerow(header_row)

    def _append_predictions_to_csv(self, predictions, file_path):
        """ Appends a row to an existing csv """
        print("Writing predictions to: %s" % file_path)

        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
            for _id, values in predictions.items():
                for label_type, preds in values.items():
                    row_to_write = [_id, label_type,
                                    preds['predicted_class'],
                                    preds['predicted_value'],
                                    preds['class_predictions']]
                csvwriter.writerow(row_to_write)

    def export_predictions_csv(self, file_path):
        """ Export predictions as CSV """

        print("Start writing file: %s" % file_path)
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
            # Write Header
            header_row = ['id', 'label', 'predicted_class',
                          'predicted_value', 'class_predictions']
            csvwriter.writerow(header_row)

            for file_name, values in self.predictions.items():
                for label_type, preds in values.items():
                    row_to_write = [file_name, label_type,
                                    preds['predicted_class'],
                                    preds['predicted_value'],
                                    preds['class_predictions']]

                csvwriter.writerow(row_to_write)

        print("Finished writing file: %s" % file_path)

    def _predict_from_iterator_and_export_to_csv(self, iterator, export_path):
        """ Generate Predictions from Dataset Iterator """

        all_predictions = OrderedDict()
        output_names = self.model.output_names
        id_to_class_mapping_clean = {'label/' + k: v for k, v in
                                     self.id_to_class_mapping.items()}
        with K.get_session() as sess:
            while True:
                try:
                    batch_data = sess.run(iterator)
                    batch_predictions = OrderedDict()
                except tf.errors.OutOfRangeError:
                    print("")
                    print("Finished Predicting")
                    break
                except Exception as error:
                    traceback.print_exc()
                    continue
                images = batch_data['images']
                ids = [x.decode('utf-8') for x in batch_data['id']]
                preds_list = self.model.predict_on_batch(images)
                if not isinstance(preds_list, list):
                    preds_list = [preds_list]

                for i, _id in enumerate(ids):
                    id_preds = [x[i, :] for x in preds_list]
                    result = dict()
                    for o, output in enumerate(output_names):
                        id_output_preds = id_preds[o]
                        class_preds = {id_to_class_mapping_clean[output][ii]: y
                                       for ii, y in enumerate(id_output_preds)}

                        ordered_classes = sorted(class_preds,
                                                 key=class_preds.get,
                                                 reverse=True)

                        top_label = ordered_classes[0]
                        top_value = class_preds[top_label]
                        result[output] = {
                           'predicted_class': top_label,
                           'predicted_value': top_value,
                           'class_predictions': class_preds}
                    all_predictions[_id] = result
                    batch_predictions[_id] = result

                # append batch predictions to export here
                self._append_predictions_to_csv(batch_predictions, export_path)

        self.predictions = all_predictions


        # self.id_to_class_mapping
        #
        # pred_results =
        #
        # with tf.Session() as sess:
        #     batch = sess.run(iterator)
        #     images = batch['images']
        #     ids = batch['id']
        #     image_paths = batch['image_paths']
        #     preds = self.model.predict_on_batch(images)
        #     batch_results = {k: dict() for k in ids}
        #
        #     for i, output in enumerate(self.model.output_names):
        #         pred_output = preds[i]
        #
        #         pred_confidence = np.max(pred_output, axis=1)
        #         pred_index = np.argmax(pred_output, axis=1)
        #         pred_mapped = [self.id_to_class_mapping[output][i]
        #                        for i in pred_index]
        #
        #         true_index = batch[output]
        #         true_mapped = [self.id_to_class_mapping[output][i[0]]
        #                        for i in true_index]
        #
        #         for jj, _id in enumerate(ids):
        #             output_res = {'true': true_mapped[jj],
        #                           'pred': pred_mapped[jj],
        #                           'conf': pred_confidence[jj]}
        #
        #             batch_results[_id][output] = output_res
