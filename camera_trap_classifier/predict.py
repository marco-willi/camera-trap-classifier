#! /usr/bin/env python
""" Classify images using a trained model

    Arguments:
    -image_dir: path to root of image directory, can contain subdirectories
        with images, the program will search for all images and predict them

    - results_file: path to the file to store the predictions in

    -model_path: path to the model to use (hdf5 file)

    -class_mapping_json: path to 'label_mappings.json'

    -pre_processing_json: path to the image_processing.json

    -export_file_type (optional, default csv): csv or json

    -batch_size (optional, default 128): the number of images once at a time
        to predict before writing results to disk

    Usage example:

    python3 predict.py -image_dir /user/images/ \
    -results_file /user/predictions/output.csv \
    -export_file_type csv \
    -model_path /user/models/my_super_model.hdf5 \
    -class_mapping_json /user/models/label_mappings.json \
    -pre_processing_json /user/models/image_processing.json

"""
import argparse

from camera_trap_classifier.predicting.predictor import Predictor


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-csv_path", type=str, required=False, default=None,
        help="path to a csv containing capture events and links to images. \
            Must contain a column with a unique id and one or multiple\
            columns with links to images for that unique id.")
    parser.add_argument(
        "-csv_images_root_path", type=str, required=False, default="",
        help="Optional root path appended to each image for images in \
              'csv_path'")
    parser.add_argument(
        "-csv_id_col", type=str, required=False, default="",
        help="Column name of the id column in the csv")
    parser.add_argument(
        "-csv_images_cols", nargs='+', type=str, default=[""],
        help="Column name of the columns with the image paths, specify like \
              this: image_col1 image_col2 image_col3")
    parser.add_argument(
        "-image_dir", type=str, required=False, default=None,
        help='path to root of image directory, can contain subdirectories \
              with images, the program will search for all images and \
              predict them')
    parser.add_argument(
        "-results_file", type=str, required=True,
        help='path to the file to which to store the predictions')
    parser.add_argument(
        "-export_file_type", type=str, default="csv",
        required=False,
        choices=['csv', 'json'],
        help='export file type - csv or json')
    parser.add_argument(
        "-model_path", type=str, required=True,
        help='path to the model to use (hdf5 file)')
    parser.add_argument(
        "-class_mapping_json", type=str, required=True,
        help="path to label_mappings.json")
    parser.add_argument(
        "-pre_processing_json", type=str, required=True,
        help="path to the image_processing.json")
    parser.add_argument(
        "-batch_size", default=128, type=int, required=False,
        help="the number of images once at a time \
              to predict before writing results to disk (default 128)")
    parser.add_argument(
        "-aggregation_mode", default='mean', type=str, required=False,
        choices=['mean', 'max', 'min'],
        help="how to aggregate multiple predictions from multiple images for \
              one capture event")

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    assert sum([x is None for x in [args['csv_path'],
                args['image_dir']]]) == 1, \
        "Only one of csv_path and image_dir can be specified"

    pred = Predictor(
        model_path=args['model_path'],
        class_mapping_json=args['class_mapping_json'],
        pre_processing_json=args['pre_processing_json'],
        aggregation_mode=args['aggregation_mode'])

    if args['image_dir'] is not None:
        pred.predict_from_image_dir(
            image_dir=args['image_dir'],
            export_type=args['export_file_type'],
            output_file=args['results_file'],
            batch_size=args['batch_size'])
    else:
        pred.predict_from_csv(
            path_to_csv=args['csv_path'],
            image_root_path=args['csv_images_root_path'],
            capture_id_col=args['csv_id_col'],
            image_path_col_list=args['csv_images_cols'],
            export_type=args['export_file_type'],
            output_file=args['results_file'],
            batch_size=args['batch_size'])


if __name__ == '__main__':
    main()
