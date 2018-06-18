""" Export a trained model to Tensorflow Serving
    Requires TF 1.9
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.estimator import model_to_estimator
from pre_processing.image_transformations import preprocess_image
from data_processing.utils import  read_json
from tensorflow.python.keras.models import (
        load_model, model_from_json)
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    tag_constants, signature_constants, signature_def_utils_impl)

# Parameters
model_path = './test_big/cats_vs_dogs_multi/model_save_dir/prediction_model.hdf5'
label_mappings = './test_big/cats_vs_dogs_multi/model_save_dir/label_mappings.json'
pre_processing = './test_big/cats_vs_dogs_multi/model_save_dir/image_processing.json'
deploy_path = './test_big/cats_vs_dogs_multi/deploy_estimator/'
estimator_path = './test_big/cats_vs_dogs_multi/estimator/'
deploy_version = 1

# Remote parameters
# git clone -b deploy_models https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier
model_path = '/host/data_hdd/ctc/ss/example/saves/prediction_model.hdf5'
label_mappings = '/host/data_hdd/ctc/ss/example/saves/label_mappings.json'
pre_processing = '/host/data_hdd/ctc/ss/example/saves/image_processing.json'
deploy_path = '/host/data_hdd/ctc/ss/example/deploy_estimator/'
estimator_path = '/host/data_hdd/ctc/ss/example/estimator/'
deploy_version = 1


keras_model = load_model(model_path)

input_names = keras_model.input_names
output_names = keras_model.output_names

label_mapping = read_json(label_mappings)
pre_processing = read_json(pre_processing)
estimator = model_to_estimator(
    keras_model_path=model_path,
    model_dir=estimator_path)


def decode_and_process_image(image):
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
    deploy_path,
    serving_input_receiver_fn=serving_input_receiver_fn,
    assets_extra={'label_mappings.json': label_mappings})
