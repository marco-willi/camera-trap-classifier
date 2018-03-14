""" Class To Import and Read Datasets """
from config.config import logging
import os
from data_processing.utils import clean_input_path
import json


class ImportFromJson(object):
    """ Read Data From Json """

    def read_from_json(self, path_to_json):
        """ Read Json File """
        data_dict = self._read_json(path_to_json)
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
            raise

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


class ImportFromImageDirs(object):
    """ Read Data From Image Directories """

    def read_from_image_root_dir(self, path_to_image_root):
        """ Create inventory from path which contains class-specific
            directories
        """
        root_path = clean_input_path(path_to_image_root)
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
