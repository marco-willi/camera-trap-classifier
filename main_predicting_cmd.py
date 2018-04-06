#! /usr/bin/env python
""" Classify images using a trained model

    Usage example:

    python3 main_predicting_cmd.py -image_dir /user/images/ \
    -results_file /user/predictions/output.csv \
    -model_path /user/models/my_super_model.hdf5 \
    -class_mapping_json /user/models/label_mappings.json \
    -pre_processing_json /user/models/image_processing.json

"""
import argparse

from predicting.predictor import Predictor

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_dir", type=str, required=True)
    parser.add_argument("-results_file", type=str, required=True)
    parser.add_argument("-model_path", type=str,required=True)
    parser.add_argument("-class_mapping_json", type=str, required=True)
    parser.add_argument("-pre_processing_json", type=str, required=True)
    parser.add_argument("-export_file_type", type=str, default="csv",
                        required=False,
                        help='export file type - only csv supported')
    parser.add_argument("-batch_size", default=128, type=int, required=False)
    parser.add_argument("-check_images", default=0, type=int, required=False)

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))


    pred = Predictor(
        model_path=args['model_path'],
        class_mapping_json=args['class_mapping_json'],
        pre_processing_json=args['pre_processing_json'],
        batch_size=args['batch_size'])

    pred.predict_image_dir_and_export(
                path_to_image_dir=args['image_dir'],
                export_file=args['results_file'],
                check_images_first=args['check_images'])


# Test command
#python main_predicting_cmd.py -image_dir D:\\Studium_GD\\Zooniverse\\CamCatProject\\data\\sample_images_raw\\ \
#-results_file D:\\Studium_GD\\Zooniverse\\CamCatProject\\data\\sample_images_raw\\preds.csv \
#-model_path D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\southern_africa\\models\\species\\model_prediction_run_201804060404_incept_res_species.hdf5 \
#-class_mapping_json D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\southern_africa\\models\\species\\label_mappings.json \
#-pre_processing_json D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\southern_africa\\models\\species\\image_processing.json \
#-batch_size 1