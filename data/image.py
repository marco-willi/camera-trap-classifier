""" Functions to handle / process images """
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from PIL import Image
import numpy as np
import io

# FLAGS
cb_distortion_range = 0.05
cr_distortion_range = 0.05


def resize_image(image, target_size):
    """ Resize Image """
    image = tf.image.resize_images(image, size=target_size)
    image = tf.divide(image, 255.0)
    return image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
    Returns:
    the cropped (and resized) image.
    Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.
    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:
    image, depths, normals = _random_crop([image, depths, normals], 120, 150)
    Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
    Returns:
    the image_list with cropped images.
    Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
            height = shape[0]
            width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
    Returns:
    the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.
    For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    Returns:
    the centered image.
    Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    means = tf.cast(means, tf.float32)

    return image - means


def _image_standardize(image, means, stdevs):
    """Subtracts the given means from each image channel.
    For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    Returns:
    the centered image.
    Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
    """

    image = _mean_image_subtraction(image, means)

    num_channels = image.get_shape().as_list()[-1]
    if len(stdevs) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    # We have a 1-D tensor of means; convert to 3-D.
    stdevs = tf.expand_dims(tf.expand_dims(stdevs, 0), 0)
    stdevs = tf.cast(stdevs, tf.float32)

    return tf.divide(image, stdevs)


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
    Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.
    Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
    Returns:
    resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         image_means,
                         image_stdevs,
                         resize_side_min,
                         resize_side_max,
                         color_augmentation):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.
    Returns:
    A preprocessed image.
    """
    resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

    image = _aspect_preserving_resize(image, resize_side)
    image = tf.random_crop(image, [output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.divide(image, tf.cast(255.0, tf.float32))

    if color_augmentation is None:
        image = _image_standardize(image, image_means, image_stdevs)
        return image

    elif color_augmentation == 'fast':
        fast_mode = True
        use_fast_color_distort = False

    elif color_augmentation == 'ultra_fast':
        fast_mode = True
        use_fast_color_distort = True

    elif color_augmentation == 'full':
        fast_mode = False
        use_fast_color_distort = False

    if use_fast_color_distort:
        image = distort_color_fast(image)
    else:
        # Randomly distort the colors. There are 4 ways to do it.
        image = apply_with_random_selector(
                    image,
                    lambda x, ordering: distort_color(x, ordering, fast_mode),
                    num_cases=4)

    image = _image_standardize(image, image_means, image_stdevs)

    return image


def preprocess_for_eval(image, output_height,
                        output_width, image_means, image_stdevs,
                        resize_side):
    """Preprocesses the given image for evaluation.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
    Returns:
    A preprocessed image.
    """
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.divide(image, tf.cast(255.0, tf.float32))
    image = _image_standardize(image, image_means, image_stdevs)

    return image


def preprocess_image(image, output_height, output_width,
                     is_training,
                     resize_side_min,
                     resize_side_max,
                     image_means=[0, 0, 0],
                     image_stdevs=[1, 1, 1],
                     color_augmentation=None):
    """Preprocesses the given image.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].
    Returns:
    A preprocessed image.
    """

    if is_training:
        return preprocess_for_train(image, output_height, output_width,
                                    image_means, image_stdevs,
                                    resize_side_min, resize_side_max,
                                    color_augmentation)
    else:
        return preprocess_for_eval(image, output_height, output_width,
                                   image_means, image_stdevs,
                                   resize_side_min)


def preprocess_image_default(image, output_height, output_width,
                             is_training,
                             resize_side_min,
                             resize_side_max,
                             image_means=[0, 0, 0],
                             image_stdevs=[1, 1, 1],
                             min_crop_size=0.8):
    """ Default Image Pre-Processing """

    # normalize between 0 and 1
    image = tf.divide(tf.cast(image, tf.float32),
                      tf.cast(255.0, tf.float32))

    if is_training:
        # Randomly flip image horizontally
        image = tf.image.random_flip_left_right(image)
        # Random crop
        rand_crop = np.random.uniform(min_crop_size, 1)
        # rand_crop = tf.random_uniform([],
        #                               minval=tf.cast(min_crop_size, tf.float32),
        #                               maxval=tf.cast(1, tf.float32),
        #                               dtype=tf.float32)
        image = tf.image.central_crop(image, rand_crop)

    image = tf.image.resize_images(
                image,
                tf.cast([output_height, output_width], tf.int32))

    # standardize
    image = _image_standardize(image, image_means, image_stdevs)

    return image


def resize_jpeg(image,  max_side):
    """ Take Raw JPEG resize with aspect ratio preservation
         and return bytes
    """
    img = Image.open(image)
    img.thumbnail([max_side, max_side], Image.ANTIALIAS)
    b = io.BytesIO()
    img.save(b, 'JPEG')
    image_bytes = b.getvalue()
    return image_bytes


def read_jpeg(image):
    """ Reads jpeg and returns Bytes """
    img = Image.open(image)
    b = io.BytesIO()
    img.save(b, 'JPEG')
    image_bytes = b.getvalue()
    return image_bytes


# https://github.com/tensorflow/tpu/blob/master/models/experimental/inception/
def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
      else:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.02)
        image = tf.image.random_contrast(image, lower=0.9, upper=1)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.9, upper=1)
        image = tf.image.random_hue(image, max_delta=0.02)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.9, upper=1)
        image = tf.image.random_hue(image, max_delta=0.02)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.02)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_contrast(image, lower=0.9, upper=1)
        image = tf.image.random_brightness(image, max_delta=0.2)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def distort_color_fast(image, scope=None):
  """Distort the color of a Tensor image.
  Distort brightness and chroma values of input image
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    br_delta = random_ops.random_uniform([], -0.2, 0.2, seed=None)
    cb_factor = random_ops.random_uniform(
        [], -cb_distortion_range, cb_distortion_range, seed=None)
    cr_factor = random_ops.random_uniform(
        [], -cr_distortion_range, cr_distortion_range, seed=None)

    channels = tf.split(axis=2, num_or_size_splits=3, value=image)
    red_offset = 1.402 * cr_factor + br_delta
    green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
    blue_offset = 1.772 * cb_factor + br_delta
    channels[0] += red_offset
    channels[1] += green_offset
    channels[2] += blue_offset
    image = tf.concat(axis=2, values=channels)
    image = tf.clip_by_value(image, 0., 1.)

    return image


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]
