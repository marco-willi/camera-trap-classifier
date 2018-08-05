""" Create a Dataset Inventory

The following sources can be used:
- dir: a directory that contains subdirectories with class specific images
- csv: a csv file that contains links and labels to images
- json: a native dataset inventory file
- panthera: a specific importer for Panthera style csvs

Example Usage:
--------------
python create_dataset_inventory.py dir -path /my_images/ \
-export_path /my_data/dataset_inventory.json
"""
import os
import argparse
import logging

from config.config_logging import setup_logging
from data.inventory import DatasetInventoryMaster


# Different functions depending on input values
def csv(args):
    """ Import From CSV """
    params = {'path': args['path'],
              'image_path_col_list': args['image_fields'],
              'capture_id_col': args['capture_id_field'],
              'attributes_col_list': args['label_fields'],
              'meta_col_list': args['meta_data_fields']}
    dinv = DatasetInventoryMaster()
    dinv.create_from_source('csv', params)
    return dinv


def json(args):
    """ Import From Json """
    params = {'path': args['path']}
    dinv = DatasetInventoryMaster()
    dinv.create_from_source('json', params)
    return dinv


def class_dir(args):
    """ Import From Class Dirs"""
    params = {'path': args['path']}
    dinv = DatasetInventoryMaster()
    dinv.create_from_source('image_dir', params)
    return dinv


def panthera(args):
    """ Import From panthera """
    params = {'path': args['path']}
    dinv = DatasetInventoryMaster()
    dinv.create_from_source('panthera', params)
    return dinv


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='CREATE DATSET INVENTORY')
    parser.add_argument(
        "-log_outdir", type=str, required=False, default=None,
        help="The directory to write logfiles to (defaults to export dir)")

    subparsers = parser.add_subparsers(help='sub-command help')

    # create parser for csv input
    parser_csv = subparsers.add_parser('csv',
                                       help='specifcy if input is a csv file')
    parser_csv.add_argument("-path", type=str, required=True,
                            help="the full path of the csv file")
    parser_csv.add_argument("-export_path", type=str, required=True,
                            help="the full path to a json file which will contain\
                                  the dataset inventory \
                                  (e.g. /my_data/dataset_inventory.json)")
    parser_csv.add_argument("-capture_id_field", type=str, required=True,
                            help="the name of the csv column with the \
                                  capture id")
    parser_csv.add_argument('-image_fields', nargs='+', type=str,
                            help='the name of the csv columns with paths to \
                                  the images (more than one possible)',
                            required=True)
    parser_csv.add_argument('-label_fields', nargs='+', type=str,
                            help='the name of the csv columns with paths to \
                                  label attributes (more than one possible)',
                            required=True)
    parser_csv.add_argument('-meta_data_fields', nargs='+', type=str,
                            help='the name of the csv columns with paths to \
                                  meta data attributes (more than one poss.)',
                            required=False)
    parser_csv.set_defaults(func=csv)

    # create parser for json input
    parser_json = subparsers.add_parser('json', help='if input is a json file')
    parser_json.add_argument("-path", type=str, required=True)
    parser_json.add_argument("-export_path", type=str, required=True,
                             help="the full path to a json file which will contain\
                                   the dataset inventory \
                                   (e.g. /my_data/dataset_inventory.json)")
    parser_json.set_defaults(func=json)

    # create parser for json input
    parser_class_dirs = subparsers.add_parser(
        'dir',
        help='if input is a directory with class directories')
    parser_class_dirs.add_argument("-path", type=str, required=True)
    parser_class_dirs.add_argument(
        "-export_path", type=str, required=True,
        help="the full path to a json file which will contain\
             the dataset inventory \
             (e.g. /my_data/dataset_inventory.json)")
    parser_class_dirs.set_defaults(func=class_dir)

    # create parser for panthera input
    parser_panthera = subparsers.add_parser(
        'panthera',
        help='if input is a panthera csv file')
    parser_panthera.add_argument("-path", type=str, required=True)
    parser_panthera.add_argument(
            "-export_path", type=str, required=True,
            help="the full path to a json file which will contain\
                 the dataset inventory \
                 (e.g. /my_data/dataset_inventory.json)")
    parser_panthera.set_defaults(func=panthera)

    # Parse command line arguments
    args = vars(parser.parse_args())

    # Configure Logging
    if args['log_outdir'] is None:
        args['log_outdir'] = os.path.split(args['export_path'])[0]
    setup_logging(log_output_path=args['log_outdir'])
    logger = logging.getLogger(__name__)

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s: %s" % (k, v))

    dinv = args['func'](args)

    dinv.log_stats()

    dinv.export_to_json(json_path=args['export_path'])
