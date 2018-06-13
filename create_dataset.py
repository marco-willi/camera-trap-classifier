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
from data_processing.data_inventory import DatasetInventoryMaster
from data_processing.data_writer import DatasetWriter
from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from pre_processing.image_transformations import resize_jpeg

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
                        default="",
                        required=False)
    parser.add_argument("-balanced_sampling_min", default=False,
                        action='store_true', required=False,
                        help="sample labels balanced to the least frequent")
    parser.add_argument("-balanced_sampling_label", type=str,
                        help='label used for balanced sampling',
                        required=False)
    parser.add_argument("-remove_label_name", type=str,
                        help='remove records with label name',
                        required=False)
    parser.add_argument("-remove_label_value", type=str,
                        help='remove records with label value',
                        required=False)
    parser.add_argument("-image_save_side_max", type=int,
                        default=500,
                        required=False,
                        help="resize image to larger side having that\
                              many pixels, typically at least 330")
    parser.add_argument("-overwrite", default=False,
                        action='store_true', required=False,
                        help="whether to overwrite existing tfr files")
    parser.add_argument("-max_records_per_file", type=int,
                        default=500000,
                        required=False,
                        help="The max number of records per TFRecord file.\
                             Multiple files are generated if the size of\
                             the dataset exceeds this value. It is recommended\
                             to use large values (default 500e3)")

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    # Create Dataset Inventory
    params = {'path': args['inventory']}
    dinv = DatasetInventoryMaster()
    dinv.create_from_source('json', params)

    # Remove records if requested
    if args['remove_label_name'] is not '':
        if args['remove_label_value'] is '':
            raise ValueError('if remove_label_name is specified\
                              remove_label_value needs to be specified')

        dinv.remove_records_with_label(label_name=args['remove_label_name'],
                                       label_value=args['remove_label_value'])

    # Log Statistics
    dinv.log_stats()

    # Determine if Meta-Column has been specified
    if args['split_by_meta']:
        splitted = dinv.split_inventory_by_meta_data_column(
                meta_colum=args['split_by_meta'])

    # Determine if balanced sampling is requested
    elif args['balanced_sampling_min']:
        if args['balanced_sampling_label'] == '':
            raise ValueError("balanced_sampling_label must be specified if \
                              balanced_sampling_min is set to true")

        splitted = dinv.split_inventory_by_random_splits_with_balanced_sample(
                split_label_min=args['balanced_sampling_label'],
                split_names=args['split_names'],
                split_percent=args['split_percent'])

    # Split without balanced sampling
    else:
        splitted = dinv.split_inventory_by_random_splits(
                split_names=args['split_names'],
                split_percent=args['split_percent'])

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
            image_pre_processing_fun=resize_jpeg,
            image_pre_processing_args={"max_side":
                                       args['image_save_side_max']},
            random_shuffle_before_save=True,
            overwrite_existing_files=args['overwrite'],
            max_records_per_file=args['max_records_per_file']
            )
