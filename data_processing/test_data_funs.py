from data_processing.data_creator import CamCatDatasetDictCreator
#from data_processing.data_reader import DatasetReader
from data_processing.tfr_encoder_decoder import CamCatTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
from data_processing.tfr_splitter import TFRecordSplitter
import tensorflow as tf
import matplotlib.pyplot as plt
from pre_processing.image_transformations import (
        preprocess_image,
        preprocess_image_default, resize_jpeg, resize_image)
import time
from PIL import Image
import numpy as np

# Create From Images Directories
test_data_image_path = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\4715\\all"
output_file = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\tests\\test.tfrecord"

dataset_dict = CamCatDatasetDictCreator()
dataset_dict.create_from_class_directories(test_data_image_path)
dataset = dataset_dict.get_dataset_dict()

tfr_encoder_decoder = CamCatTFRecordEncoderDecoder()


tfr_encoder_decoder.encode_dict_to_tfr(
        dataset, output_file,
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"max_side": 150})



tfr_encoder_decoder.encode_dict_to_tfr(
        dataset, output_file,
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"smallest_side": 200,
                                   "n_color_channels": 3})


data_reader = DatasetReader(tfr_encoder_decoder.get_tfr_decoder())


pre_proc_args = {'output_height': 150,
                 'output_width':150,
                 'image_means': [0,0,0],
                 'image_stdevs': [1,1,1],
                  'is_training':True,
                  'resize_side_min':150,
                  'resize_side_max':190}

pre_proc_args_default = {
                 'output_height': 150,
                 'output_width':150,
                 'image_means': [0,0,0],
                 'image_stdevs': [1,1,1],
                  'is_training':True,
                  'min_crop_size':0.8}




images, labels = data_reader.get_iterator(
        tfr_files=[output_file],
        batch_size=128,
        is_train=True,
        n_repeats=None,
        output_labels=["primary"],
        image_pre_processing_fun=None,
        image_pre_processing_args=None,
        max_multi_label_number=None)


with tf.Session() as sess:
    for i in range(0,10):
        img, l= sess.run([images, labels])
        print(l)
        for j in range(0, l['labels/primary'].shape[0]):
            plt.imshow(img['images'][j,:,:,:])
            plt.show()

with tf.Session() as sess:
    st = time.time()
    for i in range(0,1000):
        img, l= sess.run([images, labels])
        if (i % 100) == 0:
            elapsed = time.time() - st
            print("Required %s seconds for %s batches" % (str(elapsed), str(i)))



# Create From Images Directories
test_data_image_path = "D:\\Studium_GD\\Zooniverse\\Data\\cats_vs_dogs\\test"
test_data_image_path = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\camera_catalogue\\exp_south_africa_4"
output_file = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\tests\\test_diff_size.tfrecord"
output_path = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\tests\\"

dataset_dict = CamCatDatasetDictCreator()
dataset_dict.create_from_class_directories(test_data_image_path)
dataset = dataset_dict.get_dataset_dict()

tfr_encoder_decoder = CamCatTFRecordEncoderDecoder()

tfr_encoder_decoder.encode_dict_to_tfr(
        dataset, output_file,
        image_pre_processing_fun=resize_jpeg,
        image_pre_processing_args={"max_side": 400})

tfr_splitter = TFRecordSplitter(main_file=output_file)
tfr_splitter.split_tfr_file(output_path=output_path,
                            output_prefix="split",
                            splits=['train', 'val', 'test'],
                            split_props=[0.9,0.05,0.05])
tfr_splitter.record_numbers_per_file()

data_reader = DatasetReader(tfr_encoder_decoder.get_tfr_decoder())

pre_proc_args = {'output_height': 224,
                 'output_width':224,
                 'image_means': [0,0,0],
                 'image_stdevs': [1,1,1],
                  'is_training':True,
                  'resize_side_min':224,
                  'resize_side_max':500}

images, labels = data_reader.get_iterator(
        tfr_files=[tfr_splitter.get_splits_dict()['train']],
        batch_size=128,
        is_train=True,
        n_repeats=None,
        output_labels=["primary"],
        image_pre_processing_fun=preprocess_image_default,
        image_pre_processing_args=pre_proc_args,
        max_multi_label_number=None)


with tf.Session() as sess:
    for i in range(0,30):
        img, l= sess.run([images, labels])
        print(l)
        for j in range(0, l['labels/primary'].shape[0]):
            plt.imshow(img['images'][j,:,:,:])
            plt.show()

with tf.Session() as sess:
    st = time.time()
    for i in range(0,1000):
        img, l= sess.run([images, labels])
        if (i % 100) == 0:
            elapsed = time.time() - st
            print("Required %s seconds for %s batches" % (str(elapsed), str(i)))



# Create From Json
test_json_data = "./test/json_data_file.json"
output_file = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\tests\\test2.tfrecord"
output_path = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\tests\\"

dataset_dict = CamCatDatasetDictCreator()
dataset_dict.create_from_json(test_json_data)
dataset = dataset_dict.get_dataset_dict()

tfr_encoder_decoder = CamCatTFRecordEncoderDecoder()
tfr_encoder_decoder.encode_dict_to_tfr(dataset, output_file)

tfr_splitter = TFRecordSplitter(main_file=output_file)
tfr_splitter.split_tfr_file(output_path=output_path,
                            output_prefix="split",
                            splits=['train', 'val', 'test'],
                            split_props=[0.9,0.05,0.05])

tfr_splitter.get_split_names()
tfr_splitter.record_numbers_per_file()


data_reader = DatasetReader(tfr_encoder_decoder.get_tfr_decoder())

images, labels = data_reader.get_iterator(
        tfr_files=[output_file],
        batch_size=2,
        is_train=True,
        n_repeats=None,
        output_labels=["primary"],
        image_pre_processing_fun=None,
        max_multi_label_number=3)


with tf.Session() as sess:
    for i in range(0,10):
        img, l= sess.run([images, labels])
        print(img['images'].shape)
        print(l['labels/primary'].shape)


with tf.Session() as sess:
    for i in range(0,10):
        img, l= sess.run([images, labels])
        print(l)
        imgplot = plt.imshow(img['images'][0,:,:,:])
        plt.show()






from data_processing.utils import n_records_in_tfr
n_records_in_tfr(output_file)


image_example = "D:\\Studium_GD\\Zooniverse\\Data\\cats_vs_dogs\\train\\cat.0.jpg"
img = Image.open(image_example)
larger_output_side =700

img.thumbnail([larger_output_side, larger_output_side],Image.ANTIALIAS)

img.show()





from data_processing.tfr_encoder_decoder import DefaultTFRecordEncoderDecoder
from data_processing.data_reader import DatasetReader
path_to_tfr_output = "D:\\Studium_GD\\Zooniverse\\Data\\camtrap_trainer\\data\\4715\\"


tfr_encoder_decoder = DefaultTFRecordEncoderDecoder()

dataset_reader = DatasetReader(tfr_encoder_decoder.decode_record)

files_to_split = path_to_tfr_output + "all.tfrecord"
iterator = dataset_reader.get_iterator(
     files_to_split, batch_size=128, is_train=False, n_repeats=1,
     output_labels=['primary'],
     buffer_size=2048,
     decode_images=False,
     labels_are_numeric=False,
     max_multi_label_number=None)


with tf.Session() as sess:
    for i in range(0, 3):
        try:
            images, ids_labels = sess.run(iterator)
        except tf.errors.OutOfRangeError:
            break


