""" Test Encoding and Decoding of TFRecord """
import tensorflow as tf
import numpy
import copy

from camera_trap_classifier.data.tfr_encoder_decoder import (
    DefaultTFRecordEncoderDecoder)


class testTFREncoderDecoder(tf.test.TestCase):

    def setUp(self):
        self.coder_encoder = DefaultTFRecordEncoderDecoder()
        self.default_record = {
            'id': 'test_record', 'n_images': 1,
            'n_labels': 1,
            'image_paths': ['./test/test_images/Cats/cat0.jpg'],
            'meta_data': 'record_meta_data',
            'labelstext': 'class:cat',
            'label/class': ['cat'],
            'label/count': ['1'],
            'label_num/class': [0],
            'label_num/count': [0],
            'images': [b'IMAGEBYTES_IMAGE1']}
        self.labels = ['class', 'count']

    def testEncodingDecoding(self):
        record_data = copy.deepcopy(self.default_record)
        serialized = self.coder_encoder.encode_record(record_data)

        de_serialized = self.coder_encoder.decode_record(
                serialized,
                output_labels=self.labels,
                decode_images=False,
                return_only_ml_data=False)

        with self.test_session():
            actual = de_serialized
            for k, expected_rec in record_data.items():
                self.assertIn(k, actual)
                actual_rec = actual[k].eval()
                if isinstance(expected_rec, list):
                    expected_rec = expected_rec[0]
                if isinstance(actual_rec, numpy.ndarray):
                    actual_rec = actual_rec[0]
                if isinstance(actual_rec, bytes):
                    actual_rec = actual_rec.decode("utf-8")
                if isinstance(expected_rec, bytes):
                    expected_rec = expected_rec.decode("utf-8")
                self.assertEqual(expected_rec, actual_rec)

    def testEncodingDecodingMultiImages(self):
        record_data = copy.deepcopy(self.default_record)
        record_data['images'].append(b'IMAGEBYTES_IMAGE2')
        record_data['image_paths'].append('./test/test_images/Cats/cat1.jpg')

        serialized = self.coder_encoder.encode_record(record_data)

        de_serialized = self.coder_encoder.decode_record(
                serialized,
                output_labels=self.labels,
                decode_images=False,
                return_only_ml_data=False)

        with self.test_session():
            actual = de_serialized
            imgs = actual['images'].eval()
            self.assertEqual(imgs[0], record_data['images'][0])
            self.assertEqual(imgs[1], record_data['images'][1])
            paths = actual['image_paths'].eval()
            self.assertEqual(paths[0].decode("utf-8"), record_data['image_paths'][0])
            self.assertEqual(paths[1].decode("utf-8"), record_data['image_paths'][1])
