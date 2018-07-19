""" Predict Images using a trained Model """
import os
import json
import csv
from collections import OrderedDict
import traceback

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

from pre_processing.image_transformations import preprocess_image
from data_processing.utils import (
    print_progress, export_dict_to_json, list_pictures,
    clean_input_path)


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

    def predict_from_dataset(self, dataset, export_type, output_file):
        """  Predict from Dataset
        Args:
        - dataset: a dataset object
        - export_type: csv or json
        - output_file: path to write export file to
        """
        self._predict_dataset_and_export(
            dataset, export_type, output_file)

    def predict_from_image_dir(self, image_dir, export_type, output_file):
        """  Predict from Image Directory
        Args:
        - image_dir: path to an image directory (with potentially sub-dirs)
        - export_type: csv or json
        - output_file: path to write export file to
        """
        # from Image List
        path_to_image_dir = clean_input_path(image_dir)
        image_paths = list_pictures(path_to_image_dir,  ext='jpg|jpeg')
        n_total = len(image_paths)

        print("Found %s images in %s" %
              (len(image_paths), path_to_image_dir))

        dataset = self._create_dataset_from_paths(
            image_paths, batch_size=self.batch_size)

        self._predict_dataset_and_export(
            dataset, export_type, output_file, dataset_size=n_total)

    def _predict_dataset_and_export(self, dataset, export_type, output_file,
                                    dataset_size=None, labels=False):
        """ Calculate Predictions and Export"""

        # Prepare storage of predictions
        n_processed = 0
        output_names = self.model.output_names
        id_to_class_mapping_clean = {'label/' + k: v for k, v in
                                     self.id_to_class_mapping.items()}
        all_predictions = OrderedDict()

        # Create Dataset Iterator
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        with K.get_session() as sess:
            batch_counter = 1
            while True:
                try:
                    feat, lab = sess.run([features, labels])
                    batch_predictions = OrderedDict()
                except tf.errors.OutOfRangeError:
                    print("")
                    print("Finished Predicting")
                    break
                except Exception as error:
                    print("Error in batch %s" % batch_counter)
                    traceback.print_exc()
                    continue

                # Calculate Predictions
                images = feat['images']
                preds_list = self.model.predict_on_batch(images)

                if not isinstance(preds_list, list):
                    preds_list = [preds_list]

                # Assign ID to each Prediction
                if not isinstance(preds_list, list):
                    preds_list = [preds_list]
                if 'file_paths' in lab:
                    id_field = 'file_paths'
                elif 'id' in lab:
                    id_field = 'id'
                ids = [x.decode('utf-8') for x in lab[id_field]]

                # Process and Format each Prediction
                for i, _id in enumerate(ids):
                    id_preds = [x[i, :] for x in preds_list]
                    result = dict()
                    for o, output in enumerate(output_names):
                        output_pretty = output.strip('label/')
                        id_output_preds = id_preds[o]
                        class_preds = {id_to_class_mapping_clean[output][ii]: y
                                       for ii, y in enumerate(id_output_preds)}

                        ordered_classes = sorted(class_preds,
                                                 key=class_preds.get,
                                                 reverse=True)
                        top_label = ordered_classes[0]
                        top_value = class_preds[top_label]

                        # encode confidence scores as string for json output
                        top_value = format(top_value, '.4f')
                        for k in class_preds.keys():
                            class_preds[k] = format(class_preds[k], '.4f')

                        result[output_pretty] = {
                           'prediction_top': top_label,
                           'confidence_top': top_value,
                           'predictions_all': class_preds}

                        # add ground truth if available (for evaluations)
                        if output in lab:
                            ground_truth_num = int(lab[output][i])
                            ground_truth_str = id_to_class_mapping_clean[output][ground_truth_num]
                            result[output_pretty]['ground_truth'] = \
                                ground_truth_str

                    all_predictions[_id] = result
                    batch_predictions[_id] = result

                # append batch predictions to export here
                if export_type == 'csv':
                    if batch_counter == 1:
                        self._create_csv_file_with_header(output_file)
                    self._append_predictions_to_csv(
                        batch_predictions, output_file)
                elif export_type == 'json':
                    if batch_counter == 1:
                        self._append_predictions_to_json(
                            output_file, batch_predictions,
                            append=False)
                    else:
                        self._append_predictions_to_json(
                            output_file, batch_predictions,
                            append=True)
                else:
                    raise NotImplementedError("export type %s not implemented"
                                              % export_type)
                n_processed += images.shape[0]
                batch_counter += 1
                if dataset_size is not None:
                    print_progress(n_processed, dataset_size)
                else:
                    print("Processed %s images" % n_processed)
        if export_type == 'json':
            self._finish_json_file(output_file)

        self.predictions = all_predictions

    def _create_dataset_from_paths(self, image_paths, batch_size):
        """ Creates a dataset from image_paths """
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: self._get_and_transform_image(
                              x, self.pre_processing))
        dataset = dataset.apply(tf.contrib.data.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        return dataset

    def _get_and_transform_image(self, image_path, pre_proc_args):
        """ Returns a processed image """
        image_raw = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_raw, channels=3,
                                             try_recover_truncated=True)
        image_processed = preprocess_image(image_decoded, **pre_proc_args)
        return {'images': image_processed}, {'file_paths': image_path}

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
                          'prediction_top',
                          'confidence_top', 'predictions_all']
            csvwriter.writerow(header_row)

    def _append_predictions_to_json(self, file_path, predictions, append=False):
        """ Creates/appends data a/to json file """
        if append:
            open_mode = 'a'
            first_row = False
        else:
            open_mode = 'w'
            first_row = True

        with open(file_path, open_mode) as outfile:
            for _id, values in predictions.items():
                if first_row:
                    outfile.write('{')
                    outfile.write('"%s":' % _id)
                    json.dump(values, outfile)
                    first_row = False
                else:
                    outfile.write(',\n')
                    outfile.write('"%s":' % _id)
                    json.dump(values, outfile)

    def _finish_json_file(self, file_path):
        """ Finish Json file by appending last token """
        with open(file_path, 'a') as outfile:
            outfile.write('}')

    def _append_predictions_to_csv(self, predictions, file_path):
        """ Appends a row to an existing csv """
        print("Writing predictions to: %s" % file_path)

        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                   quoting=csv.QUOTE_ALL)
            for _id, values in predictions.items():
                for label_type, preds in values.items():
                    row_to_write = [_id, label_type,
                                    preds['prediction_top'],
                                    preds['confidence_top'],
                                    preds['predictions_all']]
                    csvwriter.writerow(row_to_write)
