""" Class To Import and Read Datasets """
import os
import json
import csv
import copy
import logging

from data_processing.utils import clean_input_path


class DatasetImporter(object):
    """ Import Dataset information from specific sources """

    subclasses = {}

    @classmethod
    def register_subclass(cls, source_type):
        def decorator(subclass):
            cls.subclasses[source_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, source_type, params):
        if source_type not in cls.subclasses:
            raise ValueError('Bad source type {}'.format(source_type))

        return cls.subclasses[source_type](params)

    def import_from_source(self):
        """ Import data from self.path and create Inventory """
        raise NotImplementedError


@DatasetImporter.register_subclass('panthera_csv')
class FromPantheraCSV(DatasetImporter):
    """ Read Data from a CSV """

    def __init__(self, path):
        self.path = path

    def import_from_source(self):
        """ Read Data From CSV """
        data_dict = self._read_csv(self.path)
        self._calc_and_log_stats(data_dict)
        return data_dict

    def _calc_and_log_stats(self, data_dict):
        """ Calc some Stats """

        # for each label type and label calculate how often it occurs
        label_stats = dict()
        for k, v in data_dict.items():
            for label_type, labels in v['labels'].items():
                if label_type not in label_stats:
                    label_stats[label_type] = dict()
                for label in labels:
                    if label not in label_stats[label_type]:
                        label_stats[label_type][label] = 0
                    label_stats[label_type][label] += 1

        # print ordered statistics
        for label_type, labels in label_stats.items():
            label_list = list()
            count_list = list()
            for label, count in labels.items():
                label_list.append(label)
                count_list.append(count)
            total_counts = sum(count_list)
            sort_index = sorted(range(len(count_list)), reverse=True,
                                key=lambda k: count_list[k])
            for idx in sort_index:
                logging.info("Label Type: %s Label: %s Records: %s (%s %%)" %
                             (label_type, label_list[idx], count_list[idx],
                              round(100 * (count_list[idx]/total_counts), 4)))

    def _consolidate(self, existing, duplicate):
        """ Consolidate records with identical id """
        # check if identical species
        consol = copy.deepcopy(existing)

        for label_type, labels in duplicate['labels'].items():
            if label_type in existing['labels']:
                labels_all = set(labels).union(set(existing['labels'][label_type]))
                consol['labels'][label_type] = list(labels_all)

        return consol

    def _categorize_counts(self, count):
        """ Categorize Counts """
        if count == -1:
            return "NA"
        elif count < 11:
            return str(count)
        elif count < 51:
            return "11-50"
        else:
            return "51+"

    def _read_csv(self, path_to_csv):
        """ Read CSV File """
        assert os.path.exists(path_to_csv), \
            "Path: %s does not exist" % path_to_csv
        data_dict = dict()
        try:
            with open(path_to_csv, 'r') as f:
                csv_reader = csv.reader(f, delimiter=',', quotechar='"')
                duplicate_count = 0
                for i, row in enumerate(csv_reader):
                    # check header
                    if i == 0:
                        if 'count' in row:
                            assert row == ['image', 'species',
                                           'count', 'survey', 'dir'], \
                                   "Header of CSV is not as expected"
                            count_in_row = True
                        else:
                            assert row == ['image', 'species',
                                           'survey', 'dir'], \
                                   "Header of CSV is not as expected"
                            count_in_row = False
                    else:
                        if count_in_row:
                            # extract fields from csv
                            _id = row[0]
                            species = row[1]
                            try:
                                species_count = int(row[2])
                            except:
                                if row[2] == 'NA':
                                    species_count = -1
                                elif row[2] == '11-50':
                                    species_count = 11
                                elif row[2] == '51+':
                                    species_count = 51
                                else:
                                    species_count = -1
                                    logging.info("Record: %s has invalid count: observed %s - saved: %s" %
                                                 (_id, row[2], species_count))
                            survey = row[3]
                            image_path = row[4]
                            species_count_cat = self._categorize_counts(species_count)
                            new_record = {'images': [image_path],
                                          'labels': {'species': [species],
                                                     'count_category': [species_count_cat]}}
                        else:
                            # extract fields from csv
                            _id = row[0]
                            species = row[1]
                            survey = row[2]
                            image_path = row[3]
                            new_record = {'images': [image_path],
                                          'labels': {'species': [species]}}

                        if species == 'NA':
                            continue

                        if _id in data_dict:
                            new_record = self._consolidate(data_dict[_id], new_record)
                            if duplicate_count < 30:
                                logging.info("ID: %s already exists - consolidating" % _id)
                                logging.info("   OLD Record:")
                                for k, v in data_dict[_id].items():
                                    logging.info("    Attr: %s - Value: %s" % (k, v))
                                logging.info("   Consolidated Record:")
                                for k, v in new_record.items():
                                    logging.info("    Attr: %s - Value: %s" % (k, v))
                            elif duplicate_count==30:
                                logging.info("More IDs already exist - consolidating all...")
                            duplicate_count += 1


                        data_dict[_id] = new_record
        except Exception as e:
            logging.error('Failed to read csv:\n' + str(e))

        return data_dict


@DatasetImporter.register_subclass('json')
class FromJson(DatasetImporter):
    """ Read Data From Json """

    def __init__(self, path):
        self.path = path

    def import_from_source(self):
        """ Read Json File """
        data_dict = self._read_json(self.path)
        data_dict_clean = self._check_and_clean_dict(data_dict)
        return data_dict_clean

    def _read_json(self, path_to_json):
        """ Read File """
        assert os.path.exists(path_to_json), \
            "Path: %s does not exist" % path_to_json

        try:
            with open(path_to_json, 'r') as f:
                data_dict = json.load(f)
        except Exception as e:
            logging.error('Failed to read Json:\n' + str(e))

        n_records = len(data_dict.keys())
        logging.info("Read %s records from %s" % (n_records, path_to_json))

        return data_dict

    def _check_and_clean_labels_entry(self, labels_dict):
        """ Check and Clean Labels Dict """

        labels_cleaned = dict()
        for label_type, label_list in labels_dict.items():
            if isinstance(label_list, list):
                labels_cleaned[label_type] = label_list

            elif isinstance(label_list, str):
                labels_cleaned[label_type] = [label_list]

            else:
                return None

            # Check for "" strings in labels
            if any(x == "" for x in labels_cleaned[label_type]):
                return None

            # check all strings in list
            if not all(isinstance(x, str) for x in labels_cleaned[label_type]):
                return None

        # Remove records without any label types
        if len(list(labels_cleaned.keys())) == 0:
            return None

        return labels_cleaned

    def _check_and_clean_images_entry(self, images_list):
        """ Check and Clean Images List """
        images_entry_cleaned = list()

        if isinstance(images_list, list):
            if not all(isinstance(x, str) for x in images_list):
                return None

            if any(x == "" for x in images_list):
                return None

            if len(images_list) == 0:
                return None

            images_entry_cleaned = images_list

        elif isinstance(images_list, str):
            images_entry_cleaned = [images_list]

        return images_entry_cleaned

    def _check_and_clean_dict(self, data_dict):
        """ Check Each Record and Clean if possible """

        dict_cleaned = dict()

        for record_id, record_values in data_dict.items():
            if not isinstance(record_values, dict):
                logging.info("Record %s has invalid data and is removed" %
                             record_id)
                continue
            # check existence of lables dictionary
            if not all([x in record_values for x in ['labels' or 'images']]):

                logging.info("Record %s has no 'labels' and/or 'images' \
                             attr and is removed" % record_id)
                continue

            record_cleaned = dict()

            # check and clean labels dict

            labels_cleaned = self._check_and_clean_labels_entry(
                                    record_values['labels'])
            if labels_cleaned is None:
                logging.info("Record %s has invalid labels entry\
                              and is removed" % record_id)
                continue

            record_cleaned['labels'] = labels_cleaned

            # check and clean images dict
            images_cleaned = self._check_and_clean_images_entry(
                                        record_values['images'])
            if images_cleaned is None:
                logging.info("Record %s has invalid images entry \
                              and is removed" % record_id)
                continue

            record_cleaned['images'] = images_cleaned

            # add record and cleaned labels to dict
            dict_cleaned[str(record_id)] = record_cleaned

        return dict_cleaned


@DatasetImporter.register_subclass('image_dir')
class FromImageDirs(DatasetImporter):
    """ Read Data From Json """

    def __init__(self, path):
        self.path = path

    def import_from_source(self):
        """ Create inventory from path which contains class-specific
            directories
        """
        root_path = clean_input_path(self.path)
        assert os.path.exists(root_path), \
            "Path: %s does not exist" % root_path

        all_records_dict = self._create_dict_from_image_folders(root_path)

        return all_records_dict

    def _check_image_path(self, root_path):
        """ Check Root Path for Class Dirs """
        # List all files in directory
        file_list = os.listdir(root_path)

        class_dir_list = \
            [x for x in file_list if os.path.isdir(os.path.join(root_path, x))]

        assert len(class_dir_list) > 0, \
            "Found no directories in %s" % root_path

        self.n_classes = len(class_dir_list)
        logging.info("Found %s classes" % self.n_classes)
        logging.debug("Found following classes %s" % class_dir_list)

        return class_dir_list

    def _create_dict_from_image_folders(self, root_path):
        """ create dictionary from image paths """

        class_dir_list = self._check_image_path(root_path)

        # Process each image and create data dictionary
        all_images_data = dict()
        for class_dir in class_dir_list:
            for image_name in os.listdir(root_path + class_dir):
                splitted_file_name = image_name.split(".")
                if len(splitted_file_name) > 2:
                    logging.info("File %s has more than one . \
                                  in filename, which is not allowed"
                                 % str(image_name))
                    continue
                unique_image_id = splitted_file_name[0]
                image_data = {
                    'images': [root_path + class_dir +
                               os.path.sep + image_name],
                    'labels': {'primary': [class_dir]}}
                all_images_data[unique_image_id] = image_data

        logging.info("Found %s images" % len(all_images_data.keys()))

        return all_images_data
