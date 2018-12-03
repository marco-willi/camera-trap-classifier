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

from predicting.predictor import Predictor


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-image_dir", type=str, required=True,
        help='path to root of image directory, can contain subdirectories \
              with images, the program will search for all images and \
              predict them')
    parser.add_argument(
        "-results_file", type=str, required=True,
        help='path to the file to which to store the predictions')
    parser.add_argument(
        "-export_file_type", type=str, default="csv",
        required=False,
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

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    pred = Predictor(
        model_path=args['model_path'],
        class_mapping_json=args['class_mapping_json'],
        pre_processing_json=args['pre_processing_json'],
        batch_size=args['batch_size'])

    pred.predict_from_image_dir(
        image_dir=args['image_dir'],
        export_type=args['export_file_type'],
        output_file=args['results_file'])


if __name__ == '__main__':
    main()
