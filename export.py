""" Export a Trained model as Tensorflow Estimator for Deployment
    with Tensorflow-Serving

    WARNING: Requires Tensorflow 1.9 or higher
"""
import argparse
import logging

import tensorflow as tf
from tensorflow.keras.estimator import model_to_estimator
from pre_processing.image_transformations import preprocess_image
from data_processing.utils import read_json
from tensorflow.python.keras.models import load_model

from config.config_logging import setup_logging
from deploy.export_tfserving_estimator import serving_input_receiver_fn

# Configure Logging
setup_logging()
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True,
                        help="path to a prediction model (.hdf5 file)")
    parser.add_argument(
        "-class_mapping_json", type=str, required=True,
        help="path to label_mappings.json")
    parser.add_argument(
        "-pre_processing_json", type=str, required=True,
        help="path to the image_processing.json")
    parser.add_argument(
        "-output_dir", type=str, required=True,
        help="Root directory to which model is exported")
    parser.add_argument(
        "-estimator_save_dir", type=str, required=False,
        help="Directory to which estimator is saved (if not specified)\
              a temporary location is chosen")

    args = vars(parser.parse_args())

    print("Using arguments:")
    for k, v in args.items():
        print("Arg: %s, Value:%s" % (k, v))

    # Load Model and extract input/output layers
    keras_model = load_model(args['model'])

    input_names = keras_model.input_names
    output_names = keras_model.output_names

    label_mapping = read_json(args['class_mapping_json'])
    pre_processing = read_json(args['pre_processing_json'])
    estimator = model_to_estimator(
        keras_model_path=args['model'],
        model_dir=args['estimator_save_dir'])

    def decode_and_process_image(image):
        """ Pre-Process a single image """"
        image = tf.image.decode_jpeg(image, channels=3)
        image = preprocess_image(image, **pre_processing)
        return image

    def serving_input_receiver_fn():
        """
        This is used to define inputs to serve the model.

        :return: ServingInputReciever
        """
        single_image = tf.placeholder(dtype=tf.string)
        receiver_tensors = {
            # The size of input image is flexible.
            'image': single_image
        }
        # decode jpeg image to tensor
        processed_image = decode_and_process_image(single_image)
        images = tf.stack([processed_image])
        # Convert give inputs to adjust to the model.
        features = {
            # Resize given images.
            input_names[0]: images
        }
        return tf.estimator.export.ServingInputReceiver(
            receiver_tensors=receiver_tensors,
            features=features)

    # Save the model
    estimator.export_savedmodel(
        args['output_dir'],
        serving_input_receiver_fn=serving_input_receiver_fn,
        assets_extra={'label_mappings.json': args['class_mapping_json']})
