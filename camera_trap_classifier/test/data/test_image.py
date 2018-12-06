""" Test Image Functions """
import tensorflow as tf

from camera_trap_classifier.data.image import (
    _mean_image_subtraction,
    _image_standardize,
    )


def create_rgb_image(vals=[0, 1, 2], shape=(2, 2)):
    a = tf.constant(vals[0], shape=shape, dtype=tf.float32)
    b = tf.constant(vals[1], shape=shape, dtype=tf.float32)
    c = tf.constant(vals[2], shape=shape, dtype=tf.float32)
    return tf.stack([a, b, c], axis=2)


class ImageStandardizationTests(tf.test.TestCase):

    def setUp(self):
        self.image = create_rgb_image([0, 1, 2])
        self.image2 = create_rgb_image([50, 100, 150])

    def testMeanSubstraction(self):
        means_to_test = [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 2],
            [2, 1, 0],
            [0.5, 0.5, 0.5]
            ]

        expected = [
            create_rgb_image([0, 1, 2]),
            create_rgb_image([-1, 0, 1]),
            create_rgb_image([0, 0, 0]),
            create_rgb_image([-2, 0, 2]),
            create_rgb_image([-0.5, 0.5, 1.5])
            ]

        with self.test_session():
            for e, m in zip(expected, means_to_test):
                actual = _mean_image_subtraction(self.image, m)
                self.assertAllEqual(actual.eval(), e)

    def testStandardizeImage(self):
        std_to_test = [
            [10, 20, 30],
            [0.5, 0.5, 0.5]
            ]

        means = [100, 100, 100]

        expected = [
            create_rgb_image([-5, 0, 50/30]),
            create_rgb_image([-100, 0, 100]),
            ]

        with self.test_session():
            with self.assertRaises(ValueError):
                _image_standardize(self.image2, means, [0, 0, 0])

            for e, s in zip(expected, std_to_test):
                actual = _image_standardize(self.image2, means, s)
                self.assertAllEqual(actual.eval(), e)


if __name__ == '__main__':
    tf.test.main()
