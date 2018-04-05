""" Test Predictor Class """
from predicting.predictor import Predictor


root_path = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\cats_and_dogs\\models\\cats_vs_dogs\\"
image_dir = 'D:\\Studium_GD\\Zooniverse\\CamCatProject\\data\\sample_images_raw\\'

pred = Predictor(
    model_path=root_path + "model_prediction_run_201804050704.hdf5",
    class_mapping_json=root_path + "label_mappings.json",
    pre_processing_json=root_path + "image_processing.json",
    batch_size=128)

pred.predict_image_dir(image_dir)



import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import backend as K

from pre_processing.image_transformations import resize_jpeg, preprocess_image_default, preprocess_image
from training.model_library import create_model


image_path = 'D:\\Studium_GD\\Zooniverse\\Data\\images_from_camera_traps\\elephant_expedition_sample_image.jpeg'
image_dir = 'D:\\Studium_GD\\Zooniverse\\CamCatProject\\data\\sample_images_raw\\'
image_name = 'rhino.JPG'
image_path = image_dir + image_name
image_paths = tf.constant([image_dir + os.path.sep + x for x in os.listdir(image_dir)])

#image_paths = tf.constant([image_path])
model_path = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\southern_africa\\experiments\\species\\models\\model_epoch_3.hdf5"


batch_size=128
pre_proc_args = {'output_height': 299, 'output_width': 299,
                 'resize_side_min': 299, 'resize_side_max': 329,
                 'image_means':[0.18326417, 0.18326417, 0.18326417],
                 'image_stdevs':[0.37113875, 0.35811457, 0.3431903],
                 'is_training': False}

def _get_and_transform_image(image_path, pre_proc_args):
    image_raw = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
    image_processed = preprocess_image(image_decoded, **pre_proc_args)
    return image_processed

dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(lambda x: _get_and_transform_image(x, pre_proc_args))
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(1)
iterator = dataset.make_one_shot_iterator()
batch = iterator.get_next()
def input_feeder_pred():
    return {'input_1': batch}


model = load_model(model_path)
pred_model = create_model(
    model_name='InceptionResNetV2',
    input_feeder=input_feeder_pred,
    target_labels=['labels/species'],
    n_classes_per_label_type=[model.output_shape[1]],
    train=False,
    test_input_shape=model.input_shape[1:])
pred_model.set_weights(model.get_weights())

pred_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

init_op = tf.global_variables_initializer()
with K.get_session() as sess:
    sess.run(init_op)
    estimator_model = tf.keras.estimator.model_to_estimator(keras_model=pred_model)
    predictions = estimator_model.predict(input_feeder_pred)
    for pred in predictions:
        print(pred)
