""" Different Data Processing and Other Helper Functions """
import sys
import os
import json
from shutil import copyfile
import re
from collections import Counter, OrderedDict
from hashlib import md5
import random
import time
from multiprocessing import Pool

import tensorflow as tf
import numpy as np


def generate_synthetic_data(**kwargs):
    """ Generate Synthetic Data """
    record = generate_synthetic_batch(**kwargs)
    dataset = tf.data.Dataset.from_tensors(record)
    dataset = dataset.repeat()
    return dataset


def generate_synthetic_batch(batch_size, image_shape, labels, n_classes,
                             n_images):
    """ Generate a synthetic data batch """
    label_dict = dict()
    # Choose a random class for each label
    for label, n_class in zip(labels, n_classes):
        random_class = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=n_class-1,
            dtype=tf.int32, name='random_label')
        label_dict[label] = random_class

    images = list()
    for i in range(n_images):
        random_image = tf.random.uniform(
            shape=(batch_size,) + image_shape,
            minval=0, maxval=255,
            dtype=tf.int32, name='random_image')
        random_image = tf.divide(random_image, 255)
        random_image = random_image - 0.5
        images.append(random_image)

    return ({'images': images[0]}, label_dict)


def map_label_list_to_numeric_dict(label_list):
    """ Map a list of labels to numeric values alphabetically
        Input: ['b', 'c', 'd', 'a']
        Output: {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    """

    numeric_map = dict()
    label_list.sort()
    for i, sorted_label in enumerate(label_list):
        numeric_map[sorted_label] = i

    return numeric_map


def order_dict_by_values(d, reversed=True):
    ordered = OrderedDict()
    for w in sorted(d, key=d.get, reverse=reversed):
        ordered[w] = d[w]
    return ordered


def _balanced_sampling(id_to_label, random_seed=123):
    """ Balanced sampling for label
        Args: id_to_label (dict), key: id, value: label
    """

    labels_all = [v for v in id_to_label.values()]
    label_stats = order_dict_by_values(Counter(labels_all), reversed=False)
    min_label = list(label_stats.keys())[0]
    min_value = label_stats[min_label]

    # assign each id to one unique class
    class_assignment = {x: list() for x in label_stats.keys()}
    remaining_record_ids = set()

    # Randomly Shuffle Ids
    all_ids = list(id_to_label.keys())
    random.seed(random_seed)
    random.shuffle(all_ids)

    for record_id in all_ids:
        label = id_to_label[record_id]
        # Add record to class assignment if label occurrence
        # is below min_value of least frequent class
        if len(class_assignment[label]) < min_value:
            class_assignment[label].append(record_id)
            remaining_record_ids.add(record_id)

    return remaining_record_ids


def _assign_zero_one_to_split(zero_one_value, split_percents, split_names):
    """ Assign a value between 0 and 1 to a split according to a percentage
        distribution
    """
    split_props_cum = [sum(split_percents[0:(i+1)]) for i in
                       range(0, len(split_percents))]
    for sn, sp in zip(split_names, split_props_cum):
        if zero_one_value <= sp:
            return sn


def randomly_split_dataset(
        split_ids,
        split_names,
        split_percent,
        balanced_sampling_min=False,
        balanced_sampling_id_to_label=None):
    """ Randomly split 'split_ids' into 'split_names' by preserving
        'split_percent' and optional balanced_sampling to min. label
        Returns dict: {'id1': 'test', 'id2': 'train'}
    """

    # Check inputs
    assert isinstance(split_names, list), "split_names must be a list"
    assert isinstance(split_percent, list), "split_percent must be a list"
    assert sum(split_percent) == 1, "split_percent must sum to 1"
    assert len(split_names) == len(split_percent), \
        "Split names must be of same length as split_percent"
    assert isinstance(balanced_sampling_min, bool), \
        "balanced_sampling_min must be a boolean"

    if balanced_sampling_min:
        assert isinstance(balanced_sampling_id_to_label, dict),\
            "balanced_sampling_id_to_label must be a dict if \
             balanced_sampling_min is specified"

    # assign each record id a split value between 0 and 1
    # derived from a hash function to ensure consistency
    # based on the capture_id
    split_vals = {x: id_to_zero_one(x) for x in split_ids}

    # assign each id into different splits based on split value
    split_assignments = list()

    split_props_cum = [sum(split_percent[0:(i+1)]) for i in
                       range(0, len(split_percent))]

    for record_id in split_ids:
        split_val = split_vals[record_id]
        for sn, sp in zip(split_names, split_props_cum):
            if split_val <= sp:
                split_assignments.append(sn)
                break

    # Balanced sampling to the minority class
    if balanced_sampling_min:
        remaining_record_ids = _balanced_sampling(balanced_sampling_id_to_label)
    else:
        remaining_record_ids = split_ids

    # create final dictionary with split assignment per record id
    final_split_assignments = dict()

    for record_id, sp in zip(split_ids, split_assignments):
        if record_id in remaining_record_ids:
            final_split_assignments[record_id] = sp

    return final_split_assignments


def slice_generator(sequence_length, n_blocks):
    """ Creates a generator to get start/end indexes for dividing a
        sequence_length into n blocks
    """
    return ((int(round((b - 1) * sequence_length/n_blocks)),
             int(round(b * sequence_length/n_blocks)))
            for b in range(1, n_blocks+1))


def estimate_remaining_time(start_time, n_total, n_current):
    """ Estimate remaining time """
    time_elapsed = time.time() - start_time
    n_remaining = n_total - (n_current - 1)
    avg_time_per_record = time_elapsed / (n_current + 1)
    estimated_time = n_remaining * avg_time_per_record
    return time.strftime("%H:%M:%S", time.gmtime(estimated_time))


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
    if not isinstance(tfr_path, list):
        tfr_path = [tfr_path]
    total = 0
    for path in tfr_path:
        total += sum(1 for _ in tf.python_io.tf_record_iterator(path))
    return total


def n_records_in_tfr_dataset(tfr_path, n_processes=4):
    """ Read the number of records in all tfr files using the Dataset API
        Input:
            list of tfr paths
        Output:
            int with number of records over all files
    """
    if not isinstance(tfr_path, list):
        tfr_path = [tfr_path]

    # Use max one process per file
    n_tfr_files = len(tfr_path)
    num_parallel_reads = min(n_processes, n_tfr_files)

    # Define a Dataset and only keep the Counter
    dataset = tf.data.TFRecordDataset(tfr_path,
                                      num_parallel_reads=num_parallel_reads)
    dataset = dataset.apply(tf.data.experimental.enumerate_dataset(start=0))
    dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                        lambda x, y: x,
                        batch_size=1024))
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    # Loop once over the whole dataset
    with tf.Session() as sess:
        while True:
            try:
                counter = sess.run(batch)
            except tf.errors.OutOfRangeError:
                break
    return counter[-1]


def n_records_in_tfr_parallel(tfr_path, n_processes=4):
    """ Read the number of records in all tfr files in parallel """
    if not isinstance(tfr_path, list):
        tfr_path = [tfr_path]
    if len(tfr_path) > 0:
        pool = Pool(processes=n_processes)
        counts = list(pool.imap_unordered(n_records_in_tfr, tfr_path))
        pool.close()
        pool.join()
        return sum(counts)
    else:
        return n_records_in_tfr(tfr_path)


def check_tfrecord_contents(path_to_tfr):
    record_iterator = tf.python_io.tf_record_iterator(path_to_tfr)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
    print(example)
    print(record)


def find_tfr_files(path, prefix=''):
    """ Find all TFR files """
    files = os.listdir(path)
    tfr_files = [x for x in files if x.endswith('.tfrecord') and
                 prefix in x]
    tfr_paths = [os.path.join(*[path, x]) for x in tfr_files]
    return tfr_paths


def find_tfr_files_pattern(path, pattern=None):
    """ Find all TFR files """
    files = os.listdir(path)
    tfr_files = [x for x in files if x.endswith('.tfrecord')]
    if pattern is None:
        pass
    # check for a single pattern
    elif isinstance(pattern, str):
        tfr_files = [x for x in tfr_files if re.search(pattern, x) is not None]
    # check for multiple patterns (AND condition)
    elif isinstance(pattern, list):
        for pat in pattern:
            tfr_files = [x for x in tfr_files if re.search(pat, x) is not None]
    tfr_paths = [os.path.join(*[path, x]) for x in tfr_files]
    return tfr_paths


def find_tfr_files_pattern_subdir(path, pattern=None):
    """ Find all TFR files (including sub-dirs)"""
    if pattern is not None:
        file_paths = [os.path.join(root, f)
                      for root, _, files in os.walk(path)
                      for f in files
                      if (f.endswith('.tfrecord') and
                          all([p in f for p in pattern]))]
    else:
        file_paths = [os.path.join(root, f)
                      for root, _, files in os.walk(path)
                      for f in files
                      if f.endswith('.tfrecord')]
    return file_paths


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


def export_dict_to_json(dict, path):
    """ Export Label Mappings to Json File """
    with open(path, 'w') as fp:
        json.dump(dict, fp)


def read_json(path_to_json):
    """ Read File """
    assert os.path.exists(path_to_json), \
        "Path: %s does not exist" % path_to_json

    try:
        with open(path_to_json, 'r') as f:
            data_dict = json.load(f)
    except Exception as e:
        raise ImportError('Failed to read Json:\n' + str(e))

    return data_dict


def id_to_zero_one(value):
    """ Deterministically assign string to value 0-1 """
    hashed = hash_string(value, constant="")
    num = assign_hash_to_zero_one(hashed)
    return num


def calc_n_batches_per_epoch(n_total, batch_size, drop_remainder=True):
    """ Calculate n batches per epoch """
    n_batches_per_epoch = n_total // batch_size
    remainder = np.min([n_total % batch_size, 1])
    if not drop_remainder:
        n_batches_per_epoch += remainder
    return int(n_batches_per_epoch)


def create_path(path, create_path=True):
    if not os.path.exists(path) & create_path:
        os.mkdir(path)
    else:
        NameError("Path %s not Found" % path)


def get_most_rescent_file_with_string(dirpath, in_str='', excl_str='!'):
    """ get most recent file from directory, that includes string """
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    b = [x for x in a if (in_str in x) and not (excl_str in x)]
    latest = b[-1]
    return dirpath + os.path.sep + latest


def find_files_with_ending(dirpath, ending):
    """ get all files in dirpath with ending """
    all = os.listdir(dirpath)
    all_files = [x for x in all if os.path.isfile(os.path.join(dirpath, x))]
    all_ending = [x for x in all_files if x.endswith(ending)]
    full_paths = [os.path.join(dirpath, x) for x in all_ending]
    return full_paths


def get_most_recent_file_from_files(files):
    """ get most recent file from files """
    files.sort(key=lambda x: os.path.getmtime(x))
    latest = files[-1]
    return latest


def copy_file(file, to):
    copyfile(file, to)


def get_file_name_from_path(file_path):
    """ Extracts file name from full path """
    return file_path.split(os.path.sep)[-1]


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    """ List jpeg/jpg images  """
    file_paths = [os.path.join(root, f)
                  for root, _, files in os.walk(directory)
                  for f in files
                  if re.match(r'([\w-]+\.(?:' + ext + '))', f,
                              re.IGNORECASE)]
    return file_paths
