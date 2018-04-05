""" Predict Images using a trained Model """
import os
import json

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

from pre_processing.image_transformations import preprocess_image


class Predictor(object):
    """ Class to Predict Images """

    def __init__(self, model_path, class_mapping_json, pre_processing_json,
                 batch_size=128):
        """ Args:
            model_path: full path to a trained Keras model
            class_mapping_json: full path to json class mapping file
            pre_processing_json: full path to pre processing json file
            batch_size: numer of images to process once at a time
        """
        self.model_path = model_path
        self.class_mapping_json = class_mapping_json
        self.pre_processing_json = pre_processing_json
        self.batch_size = batch_size
        self.class_mapping = None
        self.pre_processing = None

        # check file existence
        path_names = ['model_path', 'class_mapping_json',
                      'pre_processing_json']
        paths = [self.model_path, self.class_mapping_json,
                 self.pre_processing_json]
        for name, path in zip(path_names, paths):
            if not os.path.isfile(path):
                raise FileNotFoundError("File %s not found at %s" %
                                        (name, path))

        # read the config files
        with open(self.class_mapping_json, 'r') as json_file:
            self.class_mapping = json.load(json_file)

        with open(self.pre_processing_json, 'r') as json_file:
            self.pre_processing = json.load(json_file)

        self.model = load_model(self.model_path)

    def _predict_images(self, images_list):
        """ Calculate Predictions """

        image_files_tf = tf.constant(images_list)
        batch= self._create_dataset_iterator(image_files_tf, self.batch_size)

        predictions = dict()

        with K.get_session() as sess:
            while True:
                try:
                    batch_data = sess.run(batch)
                    images = batch_data['images']
                    file_paths = batch_data['file_path']
                    preds = self.model.predict_on_batch(images)
                    max_id = np.argmax(preds, axis=1)
                    max_val = list()
                    for i in range(0, preds.shape[0]):
                        max_val.append(preds[i, max_id[i]])
                    print("Predicted: %s with %s" % (max_id, max_val))
                except tf.errors.OutOfRangeError:
                    print("Finished Predicting")
                    break

    def predict_image_dir(self, path_to_image_dir):
        """ Args:
            - path_to_image_dir: full path to directory containing images
                to predict
        """
        image_files = [path_to_image_dir + os.path.sep + x for
                       x in os.listdir(path_to_image_dir)]
        print("Found %s images in %s" %
              (len(image_files), len(path_to_image_dir)))

        self._predict_images(image_files)

    def _create_dataset_iterator(self, image_paths, batch_size):
        """ Creates an iterator over the input images """
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: self._get_and_transform_image(
                                x, self.pre_processing))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        iterator = dataset.make_one_shot_iterator()
        images, file_path = iterator.get_next()

        return {'images': images, 'file_path': file_path}

    def _get_and_transform_image(self, image_path, pre_proc_args):
        """ Returns a processed image """
        image_raw = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
        image_processed = preprocess_image(image_decoded, **pre_proc_args)
        return image_processed, image_path
