""" Export a trained model to Tensorflow Serving """
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.estimator import model_to_estimator
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
label_mappings = './test_big/cats_vs_dogs_multi/model_save_dir/label_mapping.json'
pre_processing = './test_big/cats_vs_dogs_multi/model_save_dir/image_processing.json'
deploy_path = './test_big/cats_vs_dogs_multi/deploy_estimator/'
estimator_path = './test_big/cats_vs_dogs_multi/estimator/'
deploy_version = 1

# Remote parameters
# git clone -b deploy_models https://github.com/marco-willi/camera-trap-classifier.git ~/code/camera-trap-classifier
model_path = '/host/data_hdd/ctc/ss/example/saves/prediction_model.hdf5'
label_mappings = '/host/data_hdd/ctc/ss/example/saves/label_mapping.json'
pre_processing = '/host/data_hdd/ctc/ss/example/saves/image_processing.json'
deploy_path = '/host/data_hdd/ctc/ss/example/deploy_estimator/'
estimator_path = '/host/data_hdd/ctc/ss/example/estimator/'
deploy_version = 1

from tensorflow.keras.estimator import model_to_estimator


pre_processing = read_json(pre_processing)

estimator = model_to_estimator(keras_model_path=model_path, model_dir=estimator_path)

def create_dataset_iterator(image):
    """ Creates an iterator interating over the input images
        and applying image transformations (resizing)
    """
    dataset = tf.data.Dataset.from_tensors(image)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    images = iterator.get_next()
    return images



def decode_and_process_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image, **pre_processing)
    return image


def create_dataset_iterator(image_list):
    """ Creates an iterator interating over the input images
        and applying image transformations (resizing)
    """

    dataset = tf.data.Dataset.from_tensor_slices(image_list)
    dataset = dataset.map(lambda x: decode_and_process_image(x))
    dataset = dataset.batch(1)


    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    return images



#def _get_and_transform_image(image_path, pre_proc_args):
#    """ Returns a processed image """
#    image_raw = tf.read_file(image_path)
#    image_decoded = tf.image.decode_jpeg(image_raw, channels=3,
#                                         try_recover_truncated=True)
#    image_processed = preprocess_image(image_decoded, **pre_proc_args)
#    return image_processed, image_path

#receiver_tensor = {'image': tf.placeholder(shape=[None, 150, 150, 3], dtype=tf.float32)}

#features = {'images': tf.map_fn(preprocess_image, receiver_tensor['image'], **pre_processing)}


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.

    :return: ServingInputReciever
    """
    receiver_tensors = {
        # The size of input image is flexible.
        'image': tf.placeholder(tf.string)
    }

    image_list= tf.placeholder(dtype=tf.string)


    # decode jpeg image to tensor
    images =  decode_and_process_image(image_list)
    images = tf.stack([images])

    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        'input_3': images
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,features=features)





def serving_input_receiver_fn2():
    """
    This is used to define inputs to serve the model.

    :return: ServingInputReciever
    """
    receiver_tensors = {
        # The size of input image is flexible.
        'image': tf.placeholder(tf.string)
    }
    image_raw = tf.placeholder(dtype=tf.string)
    # decode jpeg image to tensor
    image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
    proc = preprocess_image(image_decoded, **pre_processing)
    images = create_dataset_iterator(proc)
    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        'input_3': images
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,features=features)



# Save the model
estimator.export_savedmodel(deploy_path,
                            serving_input_receiver_fn=serving_input_receiver_fn)
