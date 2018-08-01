""" Create Dataset From Dataset Inventory

This creates TFRecord files from a dataset inventory.

Exmple Usage:
--------------
python create_dataset.py -inventory ./test_big/cat_dog_dir_test.json \
-output_dir ./test_big/cats_vs_dogs/tfr_files/ \
-image_save_side_max 200 \
-split_percent 0.7 0.15 0.15 \
-overwrite

"""
import argparse
import logging

from config.config_logging import setup_logging
from data.inventory import DatasetInventoryMaster
from data.writer import DatasetWriter
from data.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data.image import resize_jpeg

# Configure Logging
setup_logging()
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='CREATE DATASET')
    parser.add_argument("-inventory", type=str, required=True,
                        help="path to inventory json file")
    parser.add_argument("-output_dir", type=str, required=True,
                        help="Directory to which TFRecord files are written")
    parser.add_argument("-split_names", nargs='+', type=str,
                        help='split dataset into these named splits',
                        default=['train', 'val', 'test'],
                        required=False)
    parser.add_argument("-split_percent", nargs='+', type=float,
                        help='split dataset into these proportions',
                        default=[0.9, 0.05, 0.05],
                        required=False)
    parser.add_argument("-split_by_meta", type=str,
                        help='split dataset by meta data field in inventory',
                        default=None,
                        required=False)
    parser.add_argument("-balanced_sampling_min", default=False,
                        action='store_true', required=False,
                        help="sample labels balanced to the least frequent\
                              value")
    parser.add_argument("-balanced_sampling_label", default=None, type=str,
                        help='label used for balanced sampling')
    parser.add_argument("-remove_label_name", nargs='+', type=str,
                        default=None,
                        help='remove records with label names (a list) and \
                              corresponding remove_label_value',
                        required=False)
    parser.add_argument("-remove_label_value", nargs='+', type=str,
                        default=None,
                        help='remove records with label value (a list) and \
                              corresponding remove_label_name',
                        required=False)
    parser.add_argument("-keep_label_name", nargs='+', type=str,
                        default=None,
                        help='keep only records with at least one of the \
                              label names (a list) and \
                              corresponding keep_label_value',
                        required=False)
    parser.add_argument("-keep_label_value", nargs='+', type=str,
                        default=None,
                        help='keep only records with label value (a list) and \
                              corresponding keep_label_name',
                        required=False)
    parser.add_argument("-remove_multi_label_records", default=True,
                        action='store_true', required=False,
                        help="whether to remove records with more than one \
                              observation (multi-label) which is not currently\
                              supported in model training")
    parser.add_argument("-image_root_path", type=str, default='',
                        help='Root path of all images - will be appended to\
                              the image paths stored in the dataset inventory',
                        required=False)
    parser.add_argument("-image_save_side_max", type=int,
                        default=500,
                        required=False,
                        help="aspect preserving resizeing of images such that\
                              the larger side of each image has that\
                              many pixels, typically at least 330\
                              (depending on the model architecture)")
    parser.add_argument("-overwrite", default=False,
                        action='store_true', required=False,
                        help="whether to overwrite existing tfr files")
    parser.add_argument("-write_tfr_in_parallel", default=False,
                        action='store_true', required=False,
                        help="whether to write tfrecords in parallel if more \
                              than one is created (preferably use \
                              'process_images_in_parallel')")
    parser.add_argument("-process_images_in_parallel", default=False,
                        action='store_true', required=False,
                        help="whether to process images in parallel \
                              (only if 'write_tfr_in_parallel' is false)")
    parser.add_argument("-process_images_in_parallel_size", type=int,
                        default=320, required=False,
                        help="if processing images in parallel - how many per \
                              process, this can influene memory requirements")
    parser.add_argument("-processes_images_in_parallel_n_processes", type=int,
                        default=4, required=False,
                        help="if processing images in parallel - how many \
                              processes to use (default 4)")
    parser.add_argument("-max_records_per_file", type=int,
                        default=5000,
                        required=False,
                        help="The max number of records per TFRecord file.\
                             Multiple files are generated if the size of\
                             the dataset exceeds this value. It is recommended\
                             to use large values (default 5000)")

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    # Create Dataset Inventory
    params = {'path': args['inventory']}
    dinv = DatasetInventoryMaster()
    dinv.create_from_source('json', params)

    # Remove multi-label subjects
    if args['remove_multi_label_records']:
        dinv.remove_multi_label_records()

    # Remove specific labels
    if args['remove_label_name'] is not None:
        if args['remove_label_value'] is None:
            raise ValueError('if remove_label_name is specified\
                              remove_label_value needs to be specified')

        dinv.remove_records_with_label(
            label_name_list=args['remove_label_name'],
            label_value_list=args['remove_label_value'])

    # keep only specific labels
    if args['keep_label_name'] is not None:
        if args['keep_label_value'] is None:
            raise ValueError('if keep_label_name is specified\
                              keep_label_value needs to be specified')

        dinv.keep_only_records_with_label(
            label_name_list=args['keep_label_name'],
            label_value_list=args['keep_label_value'])

    # Log Statistics
    dinv.log_stats()

    # Determine if Meta-Column has been specified
    if args['split_by_meta'] is not None:
        logging.debug("Splitting by metadata %s" % args['split_by_meta'])
        splitted = dinv.split_inventory_by_meta_data_column(
                meta_colum=args['split_by_meta'])

    # Determine if balanced sampling is requested
    elif args['balanced_sampling_min']:
        if args['balanced_sampling_label'] is None:
            raise ValueError("balanced_sampling_label must be specified if \
                              balanced_sampling_min is set to true")
        logging.debug("Splitting by random balanced sampling")
        splitted = dinv.split_inventory_by_random_splits_with_balanced_sample(
                split_label_min=args['balanced_sampling_label'],
                split_names=args['split_names'],
                split_percent=args['split_percent'])

    # Split without balanced sampling
    else:
        logging.debug("Splitting randomly")
        splitted = dinv.split_inventory_by_random_splits(
                split_names=args['split_names'],
                split_percent=args['split_percent'])

    # Log Statistics for different splits
    for split_name, split_data in splitted.items():
        logging.info("Stats for Split %s" % split_name)
        split_data.log_stats()

    # Write Label Mappings
    out_label_mapping = args['output_dir'] + 'label_mapping.json'
    dinv.export_label_mapping(out_label_mapping)

    # Write TFrecord files
    tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()
    tfr_writer = DatasetWriter(tfr_encoder_decoder.encode_record)

    for split_name, split_data in splitted.items():
        out_name = args['output_dir'] + split_name + '.tfrecord'
        split_data.export_to_tfrecord(
            tfr_writer,
            args['output_dir'],
            file_prefix=split_name,
            image_root_path=args['image_root_path'],
            image_pre_processing_fun=resize_jpeg,
            image_pre_processing_args={"max_side":
                                       args['image_save_side_max']},
            random_shuffle_before_save=True,
            overwrite_existing_files=args['overwrite'],
            max_records_per_file=args['max_records_per_file'],
            write_tfr_in_parallel=args['write_tfr_in_parallel'],
            process_images_in_parallel=args['process_images_in_parallel'],
            process_images_in_parallel_size=args['process_images_in_parallel_size'],
            processes_images_in_parallel_n_processes=args['processes_images_in_parallel_n_processes']
            )
