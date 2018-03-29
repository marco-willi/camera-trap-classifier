""" Class to Handle Labels and Classes of a Data Inventory """
from config.config import logging


class LabelHandler(object):
    def __init__(self, inv_data):
        self.inv_data = inv_data
        self.labels_to_numeric = None

    def get_all_label_types(self):
        """ get all label types in the dataset """
        all_label_types = set()
        for v in self.inv_data.values():
            all_label_types = all_label_types.union(v['labels'].keys())
        return all_label_types

    def get_all_labels(self):
        """ get all label types and classes """
        all_labels = dict()
        for v in self.inv_data.values():
            for cl_type, cl_list in v['labels'].items():
                if cl_type not in all_labels:
                    all_labels[cl_type] = dict()
                for cl in cl_list:
                    if cl not in all_labels[cl_type]:
                        all_labels[cl_type][cl] = 0
                    all_labels[cl_type][cl] += 1
        return all_labels

    def find_multi_label_records(self):
        """ Find records with mutliple entries in one label type """
        ids_to_remove = set()
        for record_id, record_data in self.inv_data.items():
            for labels in record_data['labels'].values():
                if len(labels) > 1:
                    ids_to_remove.add(record_id)
        return ids_to_remove

    def remove_label_types(self, label_types):
        """ Remove the specified label types """
        label_types = self._convert_to_list(label_types)
        for record_id, data in self.inv_data.items():
            for label_type in label_types:
                if label_type in data['labels']:
                    self.inv_data[record_id]['labels'].pop(label_type, None)

        self.remove_not_all_label_types_present()

    def remove_labels(self, label_type_labels):
        """ Removes the specified labels """
        assert isinstance(label_type_labels, dict), \
            "label_type_labels must be a dictionary: {'label_type': 'label'}"

        # convert all label entries to list
        for label_type, labels in label_type_labels.items():
            label_type_labels[label_type] = self._convert_to_list(labels)

        # iterate over all records
        for record_id in list(self.inv_data.keys()):
            data = self.inv_data[record_id]
            for label_type in list(data['labels'].keys()):
                labels = self.inv_data[record_id]['labels'][label_type]
                # check if label type is affected
                if label_type in label_type_labels:
                    remaining_labels = list()
                    # check if label is affected and keep only unaffected
                    for label in labels:
                        if label not in set(label_type_labels[label_type]):
                            remaining_labels.append(label)
                    if len(remaining_labels) > 0:
                        self.inv_data[record_id]['labels'][label_type] =\
                            remaining_labels
                    else:
                        self.inv_data[record_id]['labels'].pop(label_type, None)

        self.remove_not_all_label_types_present()

    def keep_only_label_types(self, label_types):
        """ Keep only the specified label types """
        label_types = set(self._convert_to_list(label_types))

        for record_id in list(self.inv_data.keys()):
            for label_type in list(self.inv_data[record_id]['labels'].keys()):
                if label_type not in label_types:
                    self.inv_data[record_id]['labels'].pop(label_type, None)

        self.remove_not_all_label_types_present()

    def keep_only_labels(self, label_type_labels):
        """ Keep only the specified labels """
        to_remove = self.get_all_labels()

        for label_type, labels in to_remove.items():
            if label_type not in label_type_labels:
                to_remove[label_type] = list(labels)
            else:
                labels_to_remove = set(to_remove[label_type].keys())
                for label in label_type_labels[label_type]:
                    if label in labels_to_remove:
                        labels_to_remove.remove(label)
                to_remove[label_type] = list(labels_to_remove)

        self.remove_labels(to_remove)

    def remove_multi_label_records(self):
        """ Removes records with mutliple entries in one label type """
        ids_to_remove = self.find_multi_label_records()
        for id_to_remove in ids_to_remove:
            self.inv_data.pop(id_to_remove, None)
        logging.info("Removed %s records with multiple labels" %
                     len(ids_to_remove))

    def _convert_to_list(self, input):
        """ Convert input to list if str, else raise error """
        if isinstance(input, list):
            return input
        elif isinstance(input, str):
            return [input]
        else:
            raise ValueError("Function input: %s has to be a list, is: %s"
                             % (input, type(input)))

    def remove_not_all_label_types_present(self):
        """ Remove Records which dont have all labels """

        # get all label types in the data inventory
        all_label_types = set()
        for record_data in self.inv_data.values():
            for label_type in record_data['labels']:
                all_label_types.add(label_type)

        # remove records which contain not all types
        ids_to_remove = list()
        for record_id, record_data in self.inv_data.items():
            for label_type in all_label_types:
                if label_type not in record_data['labels']:
                    ids_to_remove.append(record_id)
                elif not isinstance(record_data['labels'][label_type], list):
                    ids_to_remove.append(record_id)
                elif len(record_data['labels'][label_type]) == 0:
                    ids_to_remove.append(record_id)

        for i, record_to_remove in enumerate(ids_to_remove):
            if i < 10:
                logging.info(
                    "Record %s has not all label types - has:  all: %s" %
                    (record_to_remove,
                     #self.inv_data[record_to_remove]['labels'],
                     all_label_types))
            self.inv_data.pop(record_to_remove, None)

        logging.info("Removed %s records due to having not all label types" %
                     len(ids_to_remove))

    def _create_mapping_labels_to_numeric(self):
        """ Map Labels To Numerics """
        # Create inventory of all label types and labels
        # input data test: {'lalal': {'primary': ['cat'],
        #  'color':["brown", "white"]}, "1233":
        #  {"primary": ["cat", "dog"], 'color': ["purple"]}}
        # Output: {'color': {'brown': 0, 'purple': 1, 'white': 2},
        #    'primary': {'cat': 0, 'dog': 1}}

        label_dict = dict()
        for record_id, record_data in self.inv_data.items():
            label_types = record_data['labels']
            for label_type, label_list in label_types.items():
                if label_type not in label_dict:
                    label_dict[label_type] = dict()
                for label in label_list:
                    if label not in label_dict[label_type]:
                        label_dict[label_type][label] = dict()
                    label_dict[label_type][label] = ''

        # Map labels of each label type alphabetically
        for label_type, labels in label_dict.items():
            label_list = list(labels.keys())
            label_list.sort()
            for i, sorted_label in enumerate(label_list):
                label_dict[label_type][sorted_label] = i

        self.labels_to_numeric = label_dict

    def map_labels_to_numeric(self):
        """ Map Labels to Numerics """
        self._create_mapping_labels_to_numeric()
        self.map_labels(self.labels_to_numeric)

    def map_labels(self, mapping):
        """ Map existing labels to new labels """
        # Example class mapping: {'primary': {'cat':'elephant', 'dog':
        #    'giraffe'},'color':{'white':'brown', 'purple': 'brown'}}
        # Example record: {'lalal': {'primary': ['cat'],
        #     'color':["brown", "white"]}, "1233":
        #      {"primary": ["cat", "dog"], 'color': ["purple"]}}
        # Example result: {'1233': {'color': ['brown'], 'primary':
        #     ['giraffe', 'elephant']}, 'lalal':
        #      {'color': ['brown'], 'primary': ['elephant']}}

        for record_id, record_data in self.inv_data.items():
            label_types = record_data['labels']
            for label_type, label_list in label_types.items():
                new_label_list = set()
                if label_type in mapping:
                    for label in label_list:
                        if label in mapping[label_type]:
                            mapped_label = \
                                mapping[label_type][label]
                            new_label_list.add(mapped_label)
                        else:
                            new_label_list.add(label)
                    self.inv_data[record_id]['labels'][label_type] = \
                        list(new_label_list)
