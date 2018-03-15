import sys
import os
import tensorflow as tf
from hashlib import md5
import numpy as np


def print_progress(count, total):
    """ Print Progress to stdout """
    pct_complete = float(count) / total

    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write('')


def os_path_separators():
    """ list path spearators of current OS """
    seps = []
    for sep in os.path.sep, os.path.altsep:
        if sep:
            seps.append(sep)
    return seps


def wrap_int64(value):
    """ Wrap integer so it can be saved in TFRecord """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    """ Wrap bytes so it can be saved in TFRecord """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_bytes_list(value):
    """ Wrap bytes so it can be saved in TFRecord """
    return tf.train.FeatureList(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
       e.g, sentence in list of bytes
    """
    return tf.train.FeatureList(feature=[wrap_bytes(v) for v in values])


def _int64_feature_list(values):
    """Wrapper for inserting int64 FeatureList into a SequenceExample proto,
       e.g, sentence in list of bytes
    """
    return tf.train.FeatureList(feature=[wrap_int64(v) for v in values])


def _bytes_feature_list_str(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
       e.g, sentence in list of bytes
    """
    return tf.train.FeatureList(feature=[wrap_bytes(tf.compat.as_bytes(v))
                                         for v in values])


def wrap_dict_bytes_str(dic, prefix=''):
    """ Store Dictionary of Strings in TFRecord format """

    # check dictionary structure
    for v in dic.values():
        assert type(v) is str,\
            "Input dictionary does not exclusively contain " + \
            " strings - inspect json format of input file"

    dic_tf = {prefix + k: wrap_bytes(tf.compat.as_bytes(v))
              for k, v in dic.items()}
    return dic_tf


def wrap_dict_bytes_list(dic, prefix=''):
    """ Store Dictionary of Lists in TFRecord format """
    # check dictionary structure
    for v in dic.values():
        assert type(v) is list,\
            "Input dictionary does not exclusively contain" + \
            " lists - inspect json format of input file"

    dic_tf = {prefix + k: _bytes_feature_list_str(v) for k, v in dic.items()}
    return dic_tf


def wrap_dict_int64_list(dic, prefix=''):
    """ Store Dictionary of Lists in TFRecord format """
    # check dictionary structure
    for v in dic.values():
        assert type(v) is list,\
            "Input dictionary does not exclusively contain" + \
            " lists - inspect json format of input file"

    dic_tf = {prefix + k: _int64_feature_list(v) for k, v in dic.items()}
    return dic_tf


# TODO: replace with .endswith()
def clean_input_path(path):
    """ Add separator to path if missing """
    seps = os_path_separators()
    add_sep_to_end = all([path[-len(sep)] != sep for sep in seps])
    if add_sep_to_end:
        return path + os.path.sep
    else:
        return path


def rename_files_cats_dogs(path):
    files = os.listdir(path)
    for file in files:
        new_file_name = ''.join(file.split(".")[0:2]) + "." + \
                        file.split(".")[2]
        os.rename(os.path.join(path, file), os.path.join(path,  new_file_name))


def n_records_in_tfr(tfr_path):
    return sum(1 for _ in tf.python_io.tf_record_iterator(tfr_path))


def check_tfrecord_contents(path_to_tfr):
    record_iterator = tf.python_io.tf_record_iterator(path_to_tfr)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
    print(example)
    print(record)


def hash_string(value, constant=""):
    """ Return hashed value """
    to_hash = str(value) + str(constant)
    hashed = md5(to_hash.encode('ascii')).hexdigest()
    return hashed


def assign_hash_to_zero_one(value):
    """ Assign a md5 string to a value between 0 and 1 """
    assert type(value) == str
    assert len(value) == 32

    value_6_chars = value[:6]
    value_hex = int(value_6_chars, base=16)

    max_6_char_hex_value = 0xFFFFFF

    zero_one = value_hex / max_6_char_hex_value

    return zero_one


def id_to_zero_one(value):
    """ Deterministically assign string to value 0-1 """
    hashed = hash_string(value, constant="")
    num = assign_hash_to_zero_one(hashed)
    return num


def calc_n_batches_per_epoch(n_total, batch_size):
    """ Calculate n batches per epoch """
    n_batches_per_epoch = n_total // batch_size
    remainder = np.min([n_total % batch_size, 1])
    n_batches_per_epoch += remainder
    return int(n_batches_per_epoch)


# TODO: Improve
def create_default_class_mapper(all_labels, class_mapping=None):
    """ Map Classes to Integers for modelling """
    class_mapper = dict()

    if class_mapping is not None:
        class_mapping_key = dict()
        for label_type, labels_to_map in class_mapping.items():
            class_mapping_key[label_type] = dict()
            for label_to_map, label_target in labels_to_map.items():
                class_mapping_key[label_type][label_target] = {label_target: ''}
            for i, k in enumerate(class_mapping_key[label_type].keys()):
                class_mapping_key[label_type][k] = i

    # loop over all labels dictionary
    for label_type, labels in all_labels.items():

        # initialize empty key and value pairs to store final mappings
        key_id = list()
        vals = list()

        # create a dictionary for each label type
        class_mapper[label_type] = dict()

        for i, k in enumerate(labels.keys()):
            key_id.append(k)
            vals.append(i)

        class_mapper[label_type]['keys'] = key_id
        class_mapper[label_type]['values'] = vals

        # re-map if class mapping available
        if class_mapping is not None:
            for i, k in enumerate(class_mapper[label_type]['keys']):
                new_val = class_mapping_key[label_type][class_mapping[label_type][k]]
                class_mapper[label_type]['values'][i] = new_val

    return class_mapper
