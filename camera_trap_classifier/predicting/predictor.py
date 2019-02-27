""" Predict Images using a trained Model """
import os
import json
import csv
from collections import OrderedDict
import traceback
import time

import tensorflow as tf

from camera_trap_classifier.predicting.processor import ProcessPredictions
from camera_trap_classifier.training.prepare_model import load_model_from_disk
from camera_trap_classifier.data.image import (
    preprocess_image, decode_image_bytes_1D)
from camera_trap_classifier.data.utils import (
    print_progress, list_pictures, estimate_remaining_time,
    slice_generator, calc_n_batches_per_epoch)


class Predictor(object):
    """ Class to Predict Images """

    def __init__(self,
                 model_path,
                 class_mapping_json,
                 pre_processing_json,
                 aggregation_mode='mean'):
        """ Args:
            model_path: full path to a trained Keras model
            class_mapping_json: full path to json class mapping file
            pre_processing_json: full path to pre processing json file
            aggregation_mode: how to aggregate predictions from multiple images
                per capture-event
            batch_size: numer of images to process once at a time
        """
        self.model_path = model_path
        self.class_mapping_json = class_mapping_json
        self.pre_processing_json = pre_processing_json
        self.aggregation_mode = aggregation_mode
        self.class_mapping = None
        self.pre_processing = None
        self.session = tf.keras.backend.get_session()

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

        self.model = load_model_from_disk(self.model_path, compile=False)

        # Create a class to process predictions
        self.processor = ProcessPredictions(
                            model_outputs=self.model.output_names,
                            id_to_class_mapping=self.id_to_class_mapping)

    def predict_from_dataset(self, dataset, export_type, output_file):
        """  Predict from Dataset
        Args:
        - dataset: a dataset object
        - export_type: csv or json
        - output_file: path to write export file to
        """
        with self.session:
            self._predict_dataset(dataset, output_file, export_type)

    def predict_from_image_dir(self, image_dir, export_type, output_file,
                               batch_size=128):
        """  Predict from Image Directory
        Args:
        - image_dir: path to an image directory (with potentially sub-dirs)
        - export_type: csv or json
        - output_file: path to write export file to
        - batch_size: numer of images to process at the same time
        """
        image_paths = self._from_image_dir(image_dir)
        inventory = self._create_inventory_from_paths(image_paths)
        with self.session:
            self._predict_inventory(inventory, output_file,
                                    batch_size, export_type)

    def predict_from_csv(self, path_to_csv, image_root_path, capture_id_col,
                         image_path_col_list, export_type, output_file,
                         batch_size=128):
        """  Predict from Dataset
        Args:
        - path_to_csv: path to a csv file
        - image_root_path: root path of the images
        - capture_id_col: column name with capture id
        - image_path_col_list: list with columns to images pertaining to the
            specified capture_id
        - export_type: csv or json
        - output_file: path to write export file to
        - batch_size: numer of images to process at the same time
        """
        inventory = self._from_csv(path_to_csv, image_root_path,
                                   capture_id_col, image_path_col_list)
        with self.session:
            self._predict_inventory(inventory, output_file,
                                    batch_size, export_type)

    def _log_cfg(self, cfg):
        """ Print configuration file """
        for k, v in cfg.items():
            if isinstance(v, dict):
                print("  Reading values for entry: %s" % k)
                for kk, vv in v.items():
                    print("    Key: %s - Value: %s" % (kk, vv))
            else:
                print("  Key: %s - Value: %s" % (k, v))

    def _from_image_dir(self, path_to_image_dir):
        """ Find all images in a directory """
        image_paths = list_pictures(
            path_to_image_dir,
            ext=('jpg', 'jpeg', 'bmp', 'png'))
        print("Found %s images in %s" %
              (len(image_paths), path_to_image_dir))

        return image_paths

    def _create_inventory_from_paths(self, image_paths):
        """ Build a inventory of images from image_paths """
        image_paths.sort()
        inventory = OrderedDict()
        image_paths = [os.path.normpath(x) for x in image_paths]
        for path in image_paths:
            inventory[path] = {
                'images': [{'path': path}]}

        return inventory

    def _from_csv(
            self, path_to_csv, image_root_path,
            capture_id_col, image_path_col_list):
        """ Build inventory from CSV """
        # check input
        if isinstance(image_path_col_list, str):
            image_path_col_list = [image_path_col_list]

        assert isinstance(image_path_col_list, list), \
            "image_path_col must be a list"

        assert isinstance(capture_id_col, str), \
            "capture_id_col must be a string"

        assert os.path.exists(path_to_csv), \
            "Path: %s does not exist" % path_to_csv

        cols_in_csv = set(image_path_col_list).union([capture_id_col])

        inventory = OrderedDict()

        with open(path_to_csv, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            # Get and check header
            header = next(csv_reader)
            assert all([x in header for x in cols_in_csv]), \
                "CSV must have a header containing following \
                 entries: %s, found following: %s" \
                 % (cols_in_csv, header)

            # identify additional meta-data columns
            meta_cols = [x for x in header if x not in cols_in_csv]

            # map columns to position
            col_mapper = {x: header.index(x) for x in header}
            for i, row in enumerate(csv_reader):
                # extract fields from csv
                attrs = {attr: row[ind] for attr, ind in col_mapper.items()}
                capture_id = attrs[capture_id_col]
                images = [attrs[im] for im in image_path_col_list]
                images = [x for x in images if x is not '']

                # add root path to images
                images = [os.path.join(image_root_path, x) for x in images]

                # meta_data
                meta_data = {m: attrs[m] for m in meta_cols}

                # insert inventory record
                inventory[capture_id] = {
                    'images': [{'path': img, 'predictions': {}}
                               for img in images]}
                if len(meta_data) > 0:
                    inventory[capture_id]['meta_data'] = meta_data
            return inventory

    def _create_dataset_from_inventory(self, inventory, batch_size=128):
        """ Creates a dataset from inventory """

        # Create id and images path list
        ids = list()
        paths = list()
        for _id, data in inventory.items():
            # add each image to a separate iteration if image choice for sets
            # is random, to ensure all images are processed
            if self.pre_processing['image_choice_for_sets'] == 'random':
                for image in data['images']:
                    ids.append(_id)
                    paths.append([image['path']])
            # Otherwise treat images of a capture event as one unit and
            # process together
            else:
                ids.append(_id)
                paths.append([x['path'] for x in data['images']])

        # Feed ids and image paths to Dataset
        dataset = tf.data.Dataset.from_tensor_slices((ids, paths))
        dataset = dataset.map(lambda x, y: self._get_and_transform_image(
                              x, y, self.pre_processing))

        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        return dataset

    def _get_and_transform_image(self, _id, image_paths, pre_proc_args):
        """ Process a list of 1-N images """
        images_raw = tf.map_fn(
                        lambda x: tf.read_file(x),
                        image_paths, dtype=tf.string)
        # decode images
        image_decoded = decode_image_bytes_1D(images_raw, **pre_proc_args)

        # pre-process image
        image_processed = preprocess_image(image_decoded, **pre_proc_args)

        return {'images': image_processed}, \
               {'id': _id, 'image_path': image_paths}

    def _predict_dataset(self, dataset, output_file, export_type):
        """ Predict a datset """

        # iterate over the dataset
        preds = self._iterate_inventory_dataset(dataset)

        # export preds
        if export_type == 'csv':
            self._create_csv_file_with_header(output_file)
            self._append_predictions_to_csv(preds, output_file)
        elif export_type == 'json':
            self._append_predictions_to_json(
                output_file, preds,
                append=False)
        else:
            raise NotImplementedError("export type %s not implemented"
                                      % export_type)
        if export_type == 'json':
            self._finish_json_file(output_file)

    def _predict_inventory(self, inventory, output_file,
                           batch_size, export_type):
        """ Predict an inventory """
        _ids = list(inventory.keys())
        n_records = len(_ids)
        # process 5 batches at the same time before writing to disk
        n_inventories = calc_n_batches_per_epoch(
            n_records, batch_size * 5, False)
        slices = slice_generator(n_records, n_inventories)
        n_processed = 0
        start_time = time.time()

        # iterate over the dataset
        for i_start, i_end in slices:
            sub_inventory = {k: inventory[k] for k in _ids[i_start:i_end]}

            # create a dataset for the current sub-inventory
            dataset = self._create_dataset_from_inventory(
                sub_inventory, batch_size)
            sub_preds = self._iterate_inventory_dataset(dataset, sub_inventory)
            n_preds = len(sub_preds.keys())

            # export preds
            if export_type == 'csv':
                if i_start == 0:
                    meta_data = set()
                    for k, v in sub_preds.items():
                        if 'meta_data' in v:
                            meta_data = meta_data.union(v['meta_data'].keys())
                    self._create_csv_file_with_header(
                        output_file,
                        meta_headers=list(meta_data))
                self._append_predictions_to_csv(sub_preds, output_file)
            elif export_type == 'json':
                if i_start == 0:
                    self._append_predictions_to_json(
                        output_file, sub_preds,
                        append=False)
                else:
                    self._append_predictions_to_json(
                        output_file, sub_preds,
                        append=True)
            else:
                raise NotImplementedError("export type %s not implemented"
                                          % export_type)

            n_processed += n_preds
            if n_records is not None:
                print_progress(n_processed, n_records)
                estt = estimate_remaining_time(
                    start_time, n_records, n_processed)
                print("\nEstimated time remaining: %s" % estt)
            else:
                print("Processed %s images" % n_processed)

        print("\nProcessed %s records" % n_processed)

        if export_type == 'json':
            self._finish_json_file(output_file)

    def _iterate_inventory_dataset(self, dataset, inventory=None, sess=None):
        """ Iterate through dataset and feed to model """

        # Create Dataset Iterator
        iterator = dataset.make_one_shot_iterator()
        batch_data = iterator.get_next()

        # collect all labels
        inventory_predictions = dict()

        batch_counter = 0

        while True:
            try:
                features, labels = self.session.run(batch_data)
                batch_counter += 1
            except tf.errors.OutOfRangeError:
                print("")
                print("finished current iterator, predicted %s batches" %
                      batch_counter)
                break
            except Exception:
                traceback.print_exc()
                continue

            # Calculate Predictions
            batch_preds = self.model.predict_on_batch(features['images'])

            # check preds format
            if not isinstance(batch_preds, list):
                batch_preds = [batch_preds]

            n_preds = int(batch_preds[0].shape[0])

            # iterate over all preds
            for pred_id in range(0, n_preds):
                # extract data of current id
                _id = labels['id'][pred_id].decode("utf-8")
                _id_preds = [x[pred_id] for x in batch_preds]
                _id_labels = {k: v[pred_id] for k, v in labels.items()}

                # extract and map predictions
                _id_preds_extracted = \
                    self.processor.map_and_extract_model_prediction(_id_preds)

                # try to get ground truth if available
                _id_ground_truth = \
                    self.processor.map_and_extract_ground_truth(_id_labels)

                # assign a prediction to a specific image
                if inventory is not None:
                    # assign the current prediction to the correct image
                    path = labels['image_path'][pred_id][0].decode("utf-8")

                    img_id = [i for i, x in enumerate(inventory[_id]['images'])
                              if x['path'] == path]
                    img_dict = inventory[_id]['images'][img_id[0]]
                    img_dict['predictions'] = _id_preds_extracted

                    # add the predictions to the batch dictionary
                    _id = labels['id'][pred_id].decode("utf-8")
                    inventory_predictions[_id] = inventory[_id]
                else:
                    # add the predictions to the batch dictionary
                    inventory_predictions[_id] = {
                        'images': [
                            {'path': 'from_iterator',
                             'predictions': _id_preds_extracted
                             }]}

                # add ground truth
                if _id_ground_truth is not None:
                    inventory_predictions[_id]['ground_truth'] = \
                        _id_ground_truth

        # Process and aggregate predictions
        inventory_predictions = \
            self.processor.process_predictions(
                inventory_predictions, aggregation_mode=self.aggregation_mode)

        return inventory_predictions

    def _create_csv_file_with_header(self, file_path, meta_headers=[]):
            """ Creates a csv file with a header row """
            print("Creating file: %s" % file_path)

            with open(file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',',
                                       quoting=csv.QUOTE_ALL)
                # Write Header
                header_row = ['id', 'label',
                              'prediction_top',
                              'confidence_top', 'predictions_all']
                header_row += meta_headers
                csvwriter.writerow(header_row)
                self.csv_header = header_row

    def _append_predictions_to_json(self, file_path, predictions,
                                    append=False):
        """ Creates/appends data a/to json file """
        if append:
            open_mode = 'a'
            first_row = False
        else:
            open_mode = 'w'
            first_row = True

        with open(file_path, open_mode) as outfile:
            for _id, values in predictions.items():

                # convert values to strings for compatibility with json
                for label_name, preds in values['aggregated_pred'].items():
                    for label in preds.keys():
                        preds[label] = format(preds[label], '.4f')

                for label_name, conf in values['confidences_top'].items():
                    values['confidences_top'][label_name] = format(conf, '.4f')

                for image in values['images']:
                    for label_name, preds in image['predictions'].items():
                        for label in preds.keys():
                            preds[label] = format(preds[label], '.4f')
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
                for label_type, preds in values['aggregated_pred'].items():

                    preds_r = {k: format(v, '.4f') for k, v in preds.items()}
                    top_pred = values['predictions_top'][label_type]
                    top_conf = \
                        format(values['confidences_top'][label_type], '.4f')

                    data = {
                     'id': _id, 'label': label_type,
                     'prediction_top': top_pred,
                     'confidence_top': top_conf,
                     'predictions_all': preds_r}

                    # find and extract meta-data
                    if 'meta_data' in values:
                        for meta_col, meta_val in values['meta_data'].items():
                            data[meta_col] = meta_val

                    row_to_write = list()
                    for header in self.csv_header:
                        try:
                            data_element = data[header]
                        except:
                            data_element = ''
                        row_to_write.append(data_element)

                    csvwriter.writerow(row_to_write)
