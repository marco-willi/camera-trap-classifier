#! /usr/bin/env python
""" Classify images using command line program """
import argparse

from predicting.predictor import Predictor

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_dir", required=True)
    parser.add_argument("-results_file", required=True)
    parser.add_argument("-model_path", required=True)
    parser.add_argument("-class_mapping_json", required=True)
    parser.add_argument("-pre_processing_json", required=True)
    parser.add_argument("export_file_type", default="csv",
                        help='export file type - only csv supported')
    parser.add_argument("batch_size", default=128, type=int)

    args = vars(parser.parse_args())

    pred = Predictor(
        model_path=args['model_path'],
        class_mapping_json=args['class_mapping_json'],
        pre_processing_json=args['pre_processing_json'],
        batch_size=args['batch_size'])

    predictions = pred.predict_image_dir(args['image_dir'])

    if args['export_file_type'] == 'csv':
        pred.export_predictions_csv(args['results_file'])
    elif args['export_file_type'] == 'json':
        raise ValueError("JSON not supported currently")
        pred.export_predictions_json(args['results_file'])
    else:
        raise ValueError("export_file_type %s not available, choose one of %s"
                         % (args['export_file_type'], ['csv']))


# Test command
#python main_predicting_cmd.py -image_dir D:\\Studium_GD\\Zooniverse\\Data\camtrap_trainer\\data\\southern_africa\\models\\species\\ \
#-results_file D:\\Studium_GD\\Zooniverse\\CamCatProject\\data\\sample_images_raw\\preds.csv \
#-model_path D:\\Studium_GD\\Zooniverse\\Data\camtrap_trainer\\data\\southern_africa\\models\\species\\model_prediction_run_201804060404_incept_res_species.hdf5 \
#-class_mapping_json D:\\Studium_GD\\Zooniverse\\Data\camtrap_trainer\\data\\southern_africa\\models\\species\\label_mappings.json \
#-pre_processing_json D:\\Studium_GD\\Zooniverse\\Data\camtrap_trainer\\data\\southern_africa\\models\\species\\image_processing.json
