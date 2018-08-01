""" Class To Read TFRecord Files
    https://www.tensorflow.org/performance/datasets_performance
"""
import tensorflow as tf
import logging


logger = logging.getLogger(__name__)


class DatasetReader(object):
    def __init__(self, tfr_decoder):
        self.tfr_decoder = tfr_decoder

    def get_iterator(self, tfr_files, batch_size, is_train, n_repeats,
                     output_labels,
                     label_to_numeric_mapping=None,
                     buffer_size=10192, num_parallel_calls=4,
                     drop_batch_remainder=True, **kwargs):
        """ Create Iterator from TFRecord """

        assert type(output_labels) is list, "label_list must be of " + \
            " type list is of type %s" % type(output_labels)

        logger.info("Creating dataset TFR iterator")

        # Create Hash Map to map str labels to numerics if specified
        class_to_index_mappings = self._create_lookup_table(
            output_labels, label_to_numeric_mapping)

        # Create a tf.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)

        # Shuffle input files for training
        if is_train:
            dataset = dataset.shuffle(buffer_size=len(tfr_files))

        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                sloppy=is_train,
                cycle_length=12))

        dataset = dataset.prefetch(buffer_size=batch_size)

        # shuffle records only for training
        if is_train:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=buffer_size,
                    count=n_repeats))

        dataset = dataset.apply(
              tf.contrib.data.map_and_batch(
                  lambda x: self.tfr_decoder(
                          serialized_example=x,
                          output_labels=output_labels,
                          label_lookup_dict=class_to_index_mappings,
                          **kwargs),
                  batch_size=batch_size,
                  num_parallel_calls=num_parallel_calls,
                  drop_remainder=drop_batch_remainder))

        if not is_train:
            dataset = dataset.repeat(n_repeats)

        return dataset

    def _create_hash_table_from_dict(self, mapping, name=None):
        """ Create a hash table from a dictionary """
        keys, values = zip(*mapping.items())
        table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(list(keys), list(values)),
          -1, name=name)
        return table

    def _create_lookup_from_dict(self, mapping, name=None):
        """ Create a lookup table from a dictionary """
        id_to_label = {v: k for k, v in mapping.items()}
        sorted_label_names = [id_to_label[x] for x in
                              range(0, len(id_to_label))]
        table = tf.contrib.lookup.index_table_from_tensor(
                    tf.constant(sorted_label_names),
                    name=name)
        return table

    def _is_correct_mapping(self, mapping):
        """ Check label to numeric mapping is
            as expectet - input is a dict
            Example: {'cat': 0, 'dog': 1, 'doggy': 1}
        """
        all_indx_values = set()
        for k, v in mapping.items():
            all_indx_values.add(v)
        # ensure each index for 0 to N is in the indexes
        for i in range(0, len(all_indx_values)):
            if i not in all_indx_values:
                return False
        return True

    def _create_lookup_table(self, output_labels, label_to_numeric_mapping):
        """ Create a lookup table to map 'output_labels' to integers
            according to 'label_to_numeric_mapping'
        """
        if label_to_numeric_mapping is not None:
            class_to_index_mappings = dict()
            for label in output_labels:
                label_mapping = label_to_numeric_mapping[label]
                if not self._is_correct_mapping(label_mapping):
                    err_msg = "Label mapping %s is invalid" % label_mapping
                    logging.error(err_msg)
                    raise ValueError(err_msg)
                lookup_tab = self._create_hash_table_from_dict(
                    label_mapping,
                    name='%s/label_lookup' % label)
                logging.debug("Mapping labels for %s are %s" %
                              (label, label_mapping))
                class_to_index_mappings['label/%s' % label] = lookup_tab
            return class_to_index_mappings
        else:
            return None
