""" Class To Import and Read Datasets """
import os
import json
import csv
import copy
import logging

from data.utils import clean_input_path


logger = logging.getLogger(__name__)


class DatasetImporter(object):
    """ Import Dataset information from specific sources into this format
        for a specific capture event:

        - unique identifyer
        - labels list
        - meta_data dictionary (optional)
        - images paths list

    Example:
    --------

     "single_species_standard":{
       "labels": [
                    {"class": "cat", "color_brown": "1", "color_white": "0",
                     "counts": "1"}
                ],
       "meta_data": {"meta_1": "meta_data_1",
                     "meta_2": "meta_data_2"},
        "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                   "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                   "\\images\\4715\\all\\cat\\10296727_0.jpeg"
                  ]
      }
    """

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

        return cls.subclasses[source_type](**params)

    def import_from_source(cls):
        """ Import data """
        raise NotImplementedError

    def _is_labels_ok(self, labels_list):
        """ Check and Clean Labels Dict

            Expected Format:
            --------------
           "labels": [
                        {"class": "cat", "color_brown": "1", "counts": "1"}
                    ]
          """
        if not isinstance(labels_list, list):
            return False

        if len(labels_list) == 0:
            return False

        for label in labels_list:
            # entry must be a dictionary
            if not isinstance(label, dict):
                return False

            # check every entry and clean
            for attr_name, attr_val in label.items():

                # every label attribute must be a string
                if not isinstance(attr_val, str):
                    return False

                # check if there are no empty strings
                if attr_val == "":
                    return False

        return True

    def _is_images_ok(self, images_list):
        """ Check Images List

            Expected Format:
            ----------------
            "images": ["\\images\\4715\\all\\cat\\10296725_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296726_0.jpeg",
                       "\\images\\4715\\all\\cat\\10296727_0.jpeg"
                      ]
        """

        if not isinstance(images_list, list):
            return False

        # Check that images entry is a list of strings
        if isinstance(images_list, list):
            if not all(isinstance(x, str) for x in images_list):
                return False

        # Check if at least one entry contains a path
        if all(x == "" for x in images_list):
            return False

        # Check if at least one image remains
        if len(images_list) == 0:
            return False

        return True

    def _is_ok_metadata(self, meta_data):
        """ Check Meta Data Format
            Expected Format:
            -----------------
           "meta_data": {"meta_1": "meta_data_1",
                         "meta_2": "meta_data_2"}
        """
        if not isinstance(meta_data, dict):
            return False

        for k, v in meta_data.items():
            if not isinstance(v, str):
                return False

        return True

    def _remove_invalid_entries(self, data_dict):
        """ Check Each Record and Clean if possible """

        valid_data_dict = dict()
        required_record_entrys = ('labels', 'images')

        for record_id, record_values in data_dict.items():
            # Remove if any record entry is not a dictionary
            if not isinstance(record_values, dict):
                logger.debug("Record %s has invalid data and is removed" %
                             record_id)
                continue
            # check existence of required entrys
            if not all([x in record_values for x in required_record_entrys]):
                logger.debug("Record %s has not all required record entrys' \
                             and is removed" % record_id)
                continue

            # check labels
            if not self._is_labels_ok(record_values['labels']):
                logger.debug("Record %s has invalid labels entry\
                             and is removed" % record_id)
                continue

            # check images
            if not self._is_images_ok(record_values['images']):
                logger.debug("Record %s has invalid images entry \
                             and is removed" % record_id)
                continue

            # check meta_data entry
            if 'meta_data' in record_values:
                if not self._is_ok_metadata(record_values['meta_data']):
                    logger.debug("Record %s has invalid meta_data entry \
                                 and is removed" % record_id)
                    continue

            # add valid entries
            valid_data_dict[record_id] = record_values

        return valid_data_dict


@DatasetImporter.register_subclass('csv')
class FromCSV(DatasetImporter):
    """ Read Data from a CSV

    Args:
        path (str): path to csv file
        capture_id_col (str): id column of csv
        image_path_col_list (list): image columns of csv
        attributes_col_list (list): label columns of csv
        meta_col_list (list): additional attributes of csv for import
    """

    def __init__(self, path,
                 capture_id_col,
                 image_path_col_list,
                 attributes_col_list,
                 meta_col_list=None):
        self.path = path
        self.capture_id_col = capture_id_col
        self.image_path_col_list = image_path_col_list
        self.attributes_col_list = attributes_col_list
        self.meta_col_list = meta_col_list

        # check input
        if isinstance(self.image_path_col_list, str):
            self.image_path_col_list = [self.image_path_col_list]

        assert isinstance(self.image_path_col_list, list), \
            "image_path_col must be a list"

        assert isinstance(self.capture_id_col, str), \
            "capture_id_col must be a string"

        assert isinstance(self.attributes_col_list, list), \
            "attributes_col_list must be a list"

        if self.meta_col_list is not None:
            assert isinstance(self.meta_col_list, list), \
                "meta_col_list must be a list"
        else:
            self.meta_col_list = []

        self.cols_in_csv = set(self.image_path_col_list).union(
            [self.capture_id_col]).union(self.attributes_col_list).union(
            self.meta_col_list)

        self.n_images_per_capture = len(self.image_path_col_list)

    def import_from_source(self):
        """ Read Data From CSV """
        data_dict = self._read_csv(self.path)
        data_dict_clean = super()._remove_invalid_entries(data_dict)
        return data_dict_clean

    def _read_csv(self, path_to_csv):
        """ Read CSV File """
        assert os.path.exists(path_to_csv), \
            "Path: %s does not exist" % path_to_csv
        data_dict = dict()
        try:
            with open(path_to_csv, 'r') as f:
                csv_reader = csv.reader(f, delimiter=',')
                for i, row in enumerate(csv_reader):
                    # handle first row which is expected to be a header
                    if i == 0:
                        # ensure all colums are in header
                        assert all([x in self.cols_in_csv for x in row]), \
                            "CSV must have a header containing following \
                             entries: %s" % self.cols_in_csv

                        # map columns
                        col_mapper = {x: row.index(x) for x in
                                      self.cols_in_csv}
                    else:
                        # extract fields from csv
                        attrs = {attr: row[ind] for attr, ind in
                                 col_mapper.items()}

                        # build a new record
                        new_record = {}

                        # get labels
                        labels = {k: str(v) for k, v in attrs.items() if k in
                                  set(self.attributes_col_list)}

                        new_record['labels'] = [labels]

                        # get images
                        images = [attrs[im] for im in self.image_path_col_list]
                        images = [x for x in images if x is not '']

                        new_record['images'] = images

                        # get meta data
                        if len(self.meta_col_list) > 0:
                            meta = {k: str(v) for k, v in attrs.items() if k in
                                    set(self.meta_col_list)}

                            new_record['meta_data'] = meta

                        capture_id = attrs[self.capture_id_col]

                        # consolidate records
                        if capture_id in data_dict:
                            consolidated_record = self._consolidate_records(
                                first=data_dict[capture_id],
                                second=new_record
                            )

                            data_dict[capture_id] = consolidated_record
                        else:
                            data_dict[capture_id] = new_record

        except Exception as e:
            logger.error('Failed to read csv:\n' + str(e))

        return data_dict

    def _consolidate_records(self, first, second):
        """ Consolidate records with identical capture event id """
        images = first['images']
        labels = first['labels'] + second['labels']
        if 'meta_data' in first:
            return {'images': images, 'labels': labels,
                    'meta_data': first['meta_data']}
        else:
            return {'images': images, 'labels': labels}


@DatasetImporter.register_subclass('panthera_csv')
class FromPantheraCSV(DatasetImporter):
    """ Read Data from a CSV """
    def __init__(self, path, capture_id_col="image",
                 image_path_col_list=["dir"],
                 attributes_col_list=["species", "count"]):
        self.path = path
        self.capture_id_col = capture_id_col
        self.image_path_col_list = image_path_col_list
        self.attributes_col_list = attributes_col_list

        # check input
        assert isinstance(self.image_path_col_list, list), \
            "image_path_col must be a list"

        assert isinstance(self.capture_id_col, str), \
            "capture_id_col must be a string"

        assert isinstance(self.attributes_col_list, list), \
            "attributes_col_list must be a list"

        self.cols_in_csv = set(self.image_path_col_list).union(
            [self.capture_id_col]).union(self.attributes_col_list)

    def import_from_source(self):
        """ Read Data From CSV """
        data_dict = self._read_csv(self.path)
        data_dict_clean = super()._remove_invalid_entries(data_dict)
        return data_dict_clean

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
                    # handle first row which is expected to be a header
                    if i == 0:
                        # Panthera Hack
                        count_in_header = 'count' in row
                        if not count_in_header:
                            self.cols_in_csv.remove('count')

                        # ensure all colums are in header
                        assert all([x in row for x in self.cols_in_csv]), \
                            "CSV must have a header containing following \
                             entries: %s, found following: %s" \
                             % (self.cols_in_csv, row)

                        # map columns
                        col_mapper = {x: row.index(x) for x in
                                      self.cols_in_csv}

                    else:
                        # extract fields from csv
                        attrs = {attr: row[ind] for attr, ind in
                                 col_mapper.items()}

                        # build a record
                        labels = {k: str(v) for k, v in attrs.items() if k in
                                  set(self.attributes_col_list)}

                        images = [attrs[im] for im in self.image_path_col_list]

                        # handle count categories
                        if count_in_header:
                            new_count_attr = \
                                self._categorize_counts(labels['count'])
                            labels['count'] = new_count_attr

                        new_record = {'images': images,
                                      'labels': [labels]
                                      }

                        # check if capture id already exists and consolidate
                        capture_id = attrs[self.capture_id_col]
                        if capture_id in data_dict:
                            new_record = self._consolidate(
                                data_dict[capture_id],
                                new_record)

                            if duplicate_count < 30:
                                logger.info("ID: %s already exists - \
                                              consolidating" % capture_id)

                                logger.info(" OLD Record:")
                                for k, v in data_dict[capture_id].items():
                                    logger.info("  Attr: %s - Value: %s"
                                                % (k, v))
                                logger.info(" Consolidated Record:")
                                for k, v in new_record.items():
                                    logger.info("  Attr: %s - Value: %s"
                                                % (k, v))
                            elif duplicate_count == 30:
                                logger.info("More IDs already exist - \
                                              consolidating all...")
                            duplicate_count += 1

                        data_dict[capture_id] = new_record

        except Exception as e:
            logger.error('Failed to read csv:\n' + str(e))

        return data_dict

    def _consolidate(self, existing, duplicate):
        """ Consolidate records with identical id """

        consol = copy.deepcopy(existing)

        existing_labels = (x['species'] for x in existing['labels'])
        existing_images = set(existing['images'])

        for label in duplicate['labels']:
            if label['species'] not in existing_labels:
                consol['labels'].append(label)

        for image in duplicate['images']:
            if image not in existing_images:
                consol['images'].append(image)

        return consol

    def _categorize_counts(self, count):
        """ Categorize Counts """

        # try conversion to integer
        try:
            count = int(count)
        except:
            if count == 'NA':
                count = -1
            elif count == '11-50':
                count = 11
            elif count == '51+':
                count = 51
            else:
                count = -1

        # categorize
        if count == -1:
            return "NA"
        elif count < 11:
            return str(count)
        elif count < 51:
            return "11-50"
        else:
            return "51+"


@DatasetImporter.register_subclass('json')
class FromJson(DatasetImporter):
    """ Read Data From Json """

    def __init__(self, path):
        self.path = path

    def import_from_source(self):
        """ Read Json File """
        data_dict = self._read_json(self.path)
        data_dict_clean = super()._remove_invalid_entries(data_dict)
        return data_dict_clean

    def _read_json(self, path_to_json):
        """ Read File """
        assert os.path.exists(path_to_json), \
            "Path: %s does not exist" % path_to_json

        try:
            with open(path_to_json, 'r') as f:
                data_dict = json.load(f)
        except Exception as e:
            logger.error('Failed to read Json:\n' + str(e))

        n_records = len(data_dict.keys())
        logger.info("Read %s records from %s" % (n_records, path_to_json))

        return data_dict


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

        data_dict = self._create_dict_from_image_folders(root_path)
        data_dict_clean = super()._remove_invalid_entries(data_dict)
        return data_dict_clean

    def _check_image_path(self, root_path):
        """ Check Root Path for Class Dirs """
        # List all files in directory
        file_list = os.listdir(root_path)

        class_dir_list = \
            [x for x in file_list if os.path.isdir(os.path.join(root_path, x))]

        assert len(class_dir_list) > 0, \
            "Found no directories in %s" % root_path

        self.n_classes = len(class_dir_list)
        logger.info("Found %s classes" % self.n_classes)
        logger.debug("Found following classes %s" % class_dir_list)

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
                    logger.info("File %s has more than one . \
                                 in filename, which is not allowed"
                                % str(image_name))
                    continue
                unique_image_id = class_dir + '#' + splitted_file_name[0]
                image_data = {
                    'images': [root_path + class_dir +
                               os.path.sep + image_name],
                    'labels': [{'class': class_dir}]}
                all_images_data[unique_image_id] = image_data

        logger.info("Found %s images" % len(all_images_data.keys()))

        return all_images_data
