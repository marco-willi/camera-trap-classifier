""" Predict Images using a trained Model """
import os
import json
import csv
from collections import OrderedDict
import traceback

import tensorflow as tf

from camera_trap_classifier.training.prepare_model import load_model_from_disk
from camera_trap_classifier.data.image import preprocess_image
from camera_trap_classifier.data.utils import (
    print_progress, list_pictures,
    slice_generator, calc_n_batches_per_epoch)

tf.enable_eager_execution()


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

        # Analyze model output
        self.outputs = self.modelmodel.output_names
        self.output_to_pretty = \
            {x: x.split('label/')[-1] for x in self.outputs}
        self.id_to_class_mapping_clean = {'label/' + k: v for k, v in
                                          self.id_to_class_mapping.items()}

    def predict_from_dataset(self, dataset, export_type, output_file):
        """  Predict from Dataset
        Args:
        - dataset: a dataset object
        - export_type: csv or json
        - output_file: path to write export file to
        """
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
        image_paths = self._from_image_dir(self, image_dir)
        inventory = self._create_inventory_from_paths(image_paths)
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
        inventory = self._from_csv(
            self, path_to_csv, image_root_path,
            capture_id_col, image_path_col_list)

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
        image_paths = list_pictures(path_to_image_dir,
                                    ext='jpg|jpeg|bmp|png')
        print("Found %s images in %s" %
              (len(image_paths), path_to_image_dir))

        return image_paths

    def _create_inventory_from_paths(self, image_paths):
        """ Build a inventory of images from image_paths """
        image_paths.sort()
        inventory = OrderedDict()
        image_paths = [os.path.normpath(x) for x in image_paths]
        for i, path in enumerate(image_paths):
            inventory[str(i)] = {
                'images': [{'path': path}],
                'meta_data': None}

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

        cols_in_csv = set(image_path_col_list).anion(capture_id_col)

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
            col_mapper = {x: header.index(x) for x in cols_in_csv}
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
                               for img in images],
                    'meta_data': meta_data}
            return inventory

    def _create_dataset_from_inventory(self, inventory, batch_size=128):
        """ Creates a dataset from inventory """

        # Create id and images path list
        ids = list()
        paths = list()
        for _id, data in inventory.items():
            for image in data['images']:
                ids.append(_id)
                paths.append(image['path'])

        # Feed ids and image paths to Dataset
        dataset = tf.data.Dataset.from_tensor_slices((ids, paths))
        dataset = dataset.map(lambda x, y: self._get_and_transform_image(
                              x, y, self.pre_processing))

        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        return dataset

    def _get_and_transform_image(self, _id, image_path, pre_proc_args):
        """ Process a single image """
        image_raw = tf.read_file(image_path)
        image_decoded = tf.image.decode_image(image_raw, channels=3)
        image_processed = preprocess_image(image_decoded, **pre_proc_args)
        return {'images': image_processed}, \
               {'id': _id, 'image_path': image_path}

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

        for i_start, i_end in slices:
            sub_inventory = {k: inventory[k] for k in _ids[i_start:i_end]}

            # create a dataset for the current sub-inventory
            dataset = self._create_dataset_from_inventory(
                sub_inventory, batch_size)

            # iterate over the dataset
            sub_preds = self._iterate_inventory_dataset(dataset, sub_inventory)
            n_preds = len(sub_preds.keys())

            # export preds
            if export_type == 'csv':
                if i_start == 0:
                    self._create_csv_file_with_header(output_file)
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
            else:
                print("Processed %s images" % n_processed)

        if export_type == 'json':
            self._finish_json_file(output_file)

    def _iterate_inventory_dataset(self, dataset, inventory=None):
        """ Iterate through dataset and feed to model """

        # Create Dataset Iterator
        iterator = dataset.make_one_shot_iterator()

        # collect all labels
        inventory_predictions = dict()

        while True:
            try:
                features, labels = iterator.get_next()
            except tf.errors.OutOfRangeError:
                print("")
                print("Finished Predicting")
                break
            except Exception:
                traceback.print_exc()
                continue

            # Calculate Predictions
            batch_preds = self.model.predict_on_batch(features['images'])
            n_preds = int(batch_preds[0].shape[0])

            # iterate over all preds
            for pred_id in range(0, n_preds):
                # extract data of current id
                _id = labels['id'][pred_id].numpy().decode("utf-8")
                _id_preds = [x[pred_id] for x in batch_preds]
                _id_labels = {k: v[pred_id] for k, v in labels.items()}

                # extract and map predictions
                _id_preds_extracted = self._map_and_extract_model_prediction(
                     _id_preds, _id_labels, self.outputs,
                     self.output_to_pretty, self.id_to_class_mapping_clean)

                # try to get ground truth if available
                _id_ground_truth = self._map_and_extract_ground_truth(
                     _id_labels, self.outputs,
                     self.output_to_pretty, self.id_to_class_mapping_clean)

                # assign the current prediction to the correct image
                path = labels['image_path'][pred_id].numpy().decode("utf-8")

                if inventory is not None:

                    img_id = [i for i, x in enumerate(inventory[_id]['images'])
                              if x['path'] == path]
                    img_dict = inventory[_id]['images'][img_id[0]]
                    img_dict['predictions'] = _id_preds_extracted

                    # add the predictions to the batch dictionary
                    _id = labels['id'][pred_id].numpy().decode("utf-8")
                    inventory_predictions[_id] = inventory[_id]
                else:
                    # add the predictions to the batch dictionary
                    inventory_predictions[_id] = {
                        'predictions': _id_preds_extracted}

                # add ground truth
                if _id_ground_truth is not None:
                    inventory_predictions[_id]['ground_truth'] = \
                        _id_ground_truth

        # Process and aggregate predictions
        inventory_predictions = \
            self._process_predictions(inventory_predictions)

        return inventory_predictions

    def _map_and_extract_model_prediction(self, preds):
        """ Process a single prediction
        Output:
            {
                'label1': {
                    'predictions_all: {'class1': 0.5, 'class2': 0.5},
                    'prediction_top': 'class1',
                    'confidence_top': 0.5},
                'label2':
                    'predictions_all: {'classA': 0.3, 'classB': 0.7},
                    'prediction_top': 'classB',
                    'confidence_top': 0.7},
            }
        """
        result = dict()

        # Loop over all labels (self.outputs)
        for i, output in enumerate(self.outputs):

            all_numeric_outputs = \
                list(self.id_to_class_mapping_clean[output].keys())
            all_numeric_outputs.sort()

            # extract predictions for each label of the current output
            preds_for_output = preds[i]

            all_class_preds = [preds_for_output[x].numpy() for x in
                               all_numeric_outputs]

            all_class_preds_mapped = {
                    self.id_to_class_mapping_clean[output][i]: pred for i, pred
                    in enumerate(all_class_preds)
                    }

            result[self.output_to_pretty[output]] = all_class_preds_mapped

        return result

    def _map_and_extract_ground_truth(self, labels):
        """ Process a single prediction
        Output:
            {
                'label1': {
                    'predictions_all: {'class1': 0.5, 'class2': 0.5},
                    'prediction_top': 'class1',
                    'confidence_top': 0.5},
                'label2':
                    'predictions_all: {'classA': 0.3, 'classB': 0.7},
                    'prediction_top': 'classB',
                    'confidence_top': 0.7},
            }
        """
        result = dict()

        # Loop over all labels (self.outputs)
        for i, output in enumerate(self.outputs):

            truth = self._try_extracting_ground_truth(labels, output)

            if truth is not None:
                result[self.output_to_pretty[output]] = truth

        if len(result.keys()) == 0:
            return None
        return result

    def _try_extracting_ground_truth(self, labels, output):
        """ Try to find and extract ground truth """
        # try to find the output in the features
        try:
            truth_for_output = labels[output]
            try:
                truth_numeric = int(truth_for_output)
                truth_mapped = \
                    self.id_to_class_mapping_clean[output][truth_numeric]
            except:
                truth_mapped = truth_for_output.decode('utf-8')
        except:
            return None
        return truth_mapped

    def _process_predictions(self, predictions):
        """ Process a batch of predictions
        Input: {'images': [
                {'path': 'img1.jpg',
                 'predictions': {
                     'species': {'cat': '0.1', 'dog': '0.9'},
                     'standing': {'0': '0.5', '1': '0.5'}
                     },
                 {'path': 'img2.jpg',
                 'predictions': {
                     'species': {'cat': '0.3', 'dog': '0.7'},
                     'standing': {'0': '0.1', '1': '0.9'}
                     }
                 ],
            'meta_data': None}
        Output: (additional dict entry)
            ...
            aggregated_pred': {'species': {'cat': 0.4643, 'dog': 0.5357},
                    'standing': {'0': 0.5076, '1': 0.4924}}
            'predictions_top': {'species': ('dog', 0.5357),
                                'standing': ('0', 0.5076)}
            ...
        """
        for _id, data in predictions.items():
            preds_list = [x['predictions'] for x in data['images']]
            # collect all predictions
            collected = self._collect_predictions(preds_list)
            # consolidate predictions
            consolidated = self._consolidate_predictions(collected)
            # aggregate predictions
            aggregated = self._aggregate_predictions(
                consolidated,
                self.aggregation_mode)
            # get top predictions
            top_preds = self._get_top_predictions(aggregated)

            # add info
            data['aggregated_pred'] = aggregated
            data['predictions_top'] = top_preds

        return predictions

    def _collect_predictions(self, extracted_predictions):
        """ Collect all predictions
        Input:
            [{{'species': {'cat': '0.1', 'dog': '0.9'},
               'standing': {'0': '0.5', '1': '0.5'}},
            {'species': {'cat': '0.3', 'dog': '0.7'},
             'standing': {'0': '0.1', '1': '0.9'}}]
        Output:
            ['species': {'cat': ['0.1', '0.3'], 'dog': ['0.9', '0.7' ]},
            ....]
        """
        for i, image_pred in enumerate(extracted_predictions):
            all_label_names = image_pred.keys()
            if i == 0:
                preds_per_label = {x: list() for x in all_label_names}
            for label in all_label_names:
                preds_of_image_and_label = image_pred[label]
                preds_per_label[label].append(preds_of_image_and_label)
        return preds_per_label

    def _consolidate_predictions(self, collected_preds):
        """ Consolidate Predictions """

        label_names = collected_preds.keys()
        consolidated = {k: {} for k in label_names}
        for label_name in label_names:
            label_pred_list = collected_preds[label_name]
            for label_pred in label_pred_list:
                for label, pred in label_pred.items():
                    if label not in consolidated[label_name]:
                        consolidated[label_name][label] = list()
                    consolidated[label_name][label].append(pred)
        return consolidated

    def _aggregate_predictions(self, consolidated_predictions, mode='mean'):
        """ Aggregate Predictions of Multiple Images / ID

        """
        agg_label = dict()
        for label_name, labels in consolidated_predictions.items():
            agg_label[label_name] = dict()
            for label, preds_list in labels.items():
                if mode == 'mean':
                    agg = sum([float(x) for x in preds_list]) / len(preds_list)
                elif mode == 'max':
                    agg = max([float(x) for x in preds_list])
                elif mode == 'min':
                    agg = min([float(x) for x in preds_list])
                else:
                    raise NotImplementedError(
                        "Aggregation mode %s not implemented" % mode)
                agg_label[label_name][label] = agg
        return agg_label

    def _get_top_predictions(self, aggregated_predictions):
        """ Get top prediction for each label """
        top_preds = dict()
        for label_name, label_vals in aggregated_predictions.items():

            ordered_classes = sorted(label_vals,
                                     key=label_vals.get,
                                     reverse=True)
            top_label = ordered_classes[0]
            top_value = label_vals[top_label]
            top_preds[label_name] = (top_label, top_value)
        return top_preds

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

                # convert values to strings
                for label_name, preds in values['aggregated_pred'].items():
                    for label in preds.keys():
                        preds[label] = format(preds[label], '.4f')

                for label_name, preds in values['predictions_top'].items():
                    values['predictions_top'][label_name] = \
                        (preds[0], format(preds[1], '.4f'))

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
                    top_pred = values['predictions_top'][label_type][0]
                    top_conf = \
                        format(values['predictions_top'][label_type][1], '.4f')

                    row_to_write = [_id, label_type, top_pred,
                                    top_conf, preds_r]
                    csvwriter.writerow(row_to_write)
