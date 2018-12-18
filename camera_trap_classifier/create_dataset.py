""" Create Dataset From Dataset Inventory

This creates TFRecord files from a dataset inventory.

Exmple Usage:
--------------
python create_dataset.py -inventory ./test_big/cat_dog_dir_test.json \
-output_dir ./test_big/cats_vs_dogs/tfr_files/ \
-image_save_side_smallest 200 \
-split_percent 0.7 0.15 0.15 \
-overwrite

"""
import argparse
import logging

from camera_trap_classifier.config.logging import setup_logging
from camera_trap_classifier.data.inventory import DatasetInventoryMaster
from camera_trap_classifier.data.writer import DatasetWriter
from camera_trap_classifier.data.tfr_encoder_decoder import (
    DefaultTFRecordEncoderDecoder)
from camera_trap_classifier.data.image import (
    read_image_from_disk_resize_and_convert_to_jpeg)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='CREATE DATASET')
    parser.add_argument("-inventory", type=str, required=True,
                        help="path to inventory json file")
    parser.add_argument("-output_dir", type=str, required=True,
                        help="Directory to which TFRecord files are written")
    parser.add_argument(
        "-log_outdir", type=str, required=False, default=None,
        help="Directory to write logfiles to (defaults to output_dir)")
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
    parser.add_argument("-image_root_path", type=str, default=None,
                        help='Root path of all images - will be appended to\
                              the image paths stored in the dataset inventory',
                        required=False)
    parser.add_argument("-image_save_side_smallest", type=int,
                        default=500,
                        required=False,
                        help="aspect preserving resizeing of images such that\
                              the smaller side of each image has that\
                              many pixels, typically at least 330\
                              (depending on the model architecture)")
    parser.add_argument("-image_save_quality", type=int,
                        default=90,
                        choices=range(0, 101),
                        metavar="[0-100]",
                        required=False,
                        help="The image quality of the images saved to\
                              TFRecord files. Recommended is 75-90 for good\
                              quality-size trade-off.")
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

    # Parse command line arguments
    args = vars(parser.parse_args())

    # Configure Logging
    if args['log_outdir'] is None:
        args['log_outdir'] = args['output_dir']

    setup_logging(log_output_path=args['log_outdir'])

    logger = logging.getLogger(__name__)

    logger.info("Using arguments:")
    for k, v in args.items():
        logger.info("Arg: %s: %s" % (k, v))

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
        logger.debug("Splitting by metadata %s" % args['split_by_meta'])
        if args['balanced_sampling_min']:
            splitted = dinv.split_inventory_by_meta_data_column_and_balanced_sampling(
                meta_colum=args['split_by_meta'],
                split_label_min=args['balanced_sampling_label'])
            logger.debug("Balanced sampling using %s" % args['split_by_meta'])
        else:
            splitted = dinv.split_inventory_by_meta_data_column(
                    meta_colum=args['split_by_meta'])

    # Determine if balanced sampling is requested
    elif args['balanced_sampling_min']:
        if args['balanced_sampling_label'] is None:
            raise ValueError("balanced_sampling_label must be specified if \
                              balanced_sampling_min is set to true")
        logger.debug("Splitting by random balanced sampling")
        splitted = dinv.split_inventory_by_random_splits_with_balanced_sample(
                split_label_min=args['balanced_sampling_label'],
                split_names=args['split_names'],
                split_percent=args['split_percent'])

    # Split without balanced sampling
    else:
        logger.debug("Splitting randomly")
        splitted = dinv.split_inventory_by_random_splits(
                split_names=args['split_names'],
                split_percent=args['split_percent'])

    # Log all the splits to create
    for i, split_name in enumerate(splitted.keys()):
        logger.info("Created split %s - %s" % (i, split_name))

    # Log Statistics for different splits
    for split_name, split_data in splitted.items():
        logger.debug("Stats for Split %s" % split_name)
        split_data.log_stats(debug_only=True)

    # Write Label Mappings
    out_label_mapping = args['output_dir'] + 'label_mapping.json'
    dinv.export_label_mapping(out_label_mapping)

    # Write TFrecord files
    tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()
    tfr_writer = DatasetWriter(tfr_encoder_decoder.encode_record)

    counter = 0
    n_splits = len(splitted.keys())
    for split_name, split_data in splitted.items():
        counter += 1
        logger.info("Starting to process %s (%s / %s)" %
                    (split_name, counter, n_splits))
        split_data.export_to_tfrecord(
            tfr_writer,
            args['output_dir'],
            file_prefix=split_name,
            image_root_path=args['image_root_path'],
            image_pre_processing_fun=read_image_from_disk_resize_and_convert_to_jpeg,
            image_pre_processing_args={"smallest_side":
                                       args['image_save_side_smallest'],
                                       "image_save_quality":
                                       args['image_save_quality']},
            random_shuffle_before_save=True,
            overwrite_existing_files=args['overwrite'],
            max_records_per_file=args['max_records_per_file'],
            write_tfr_in_parallel=args['write_tfr_in_parallel'],
            process_images_in_parallel=args['process_images_in_parallel'],
            process_images_in_parallel_size=args['process_images_in_parallel_size'],
            processes_images_in_parallel_n_processes=args['processes_images_in_parallel_n_processes']
            )
    logger.info("Finished writing TFRecords")


if __name__ == '__main__':
    main()
