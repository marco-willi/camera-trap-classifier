""" Functions to handle / process images """
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops

# FLAGS
cb_distortion_range = 0.05
cr_distortion_range = 0.05


def read_image_from_disk_and_convert_to_jpeg(
        path_to_image,
        image_save_quality=75):
    """ TF-Functions to read and convert jpeg
        Requires tf.enable_eager_execution()
    """
    with tf.gfile.GFile(path_to_image, 'rb') as f:
        file_bytes = f.read()
    image = tf.image.decode_image(file_bytes)
    jpeg = tf.image.encode_jpeg(image, quality=image_save_quality).numpy()
    return jpeg


def read_image_from_disk_resize_and_convert_to_jpeg(
        path_to_image,
        smallest_side,
        image_save_quality=75):
    """ TF-Functions to read and convert jpeg
        Requires tf.enable_eager_execution()
    """
    with tf.gfile.GFile(path_to_image, 'rb') as f:
        file_bytes = f.read()
    image = tf.image.decode_image(file_bytes)
    image = _aspect_preserving_resize(image, smallest_side)
    image = tf.cast(image, dtype=tf.uint8)
    jpeg = tf.image.encode_jpeg(image, quality=image_save_quality).numpy()
    return jpeg


def resize_image(image, target_size):
    """ Resize Image """
    image = tf.image.resize_images(image, size=target_size)
    image = tf.divide(image, 255.0)
    return image


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

        offset_height = tf.cast((image_height - crop_height) / 2, tf.int32)
        offset_width = tf.cast((image_width - crop_width) / 2, tf.int32)

        outputs.append(tf.image.crop_to_bounding_box(
                             image, offset_height, offset_width,
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

    # check for zero stdev
    if any([x == 0 for x in stdevs]):
        raise ValueError('stdev: %s is zero, leads to div by zero' % stdevs)

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
    new_height, new_width = \
        _smallest_size_at_least(height, width, smallest_side)
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
                         color_augmentation,
                         ignore_aspect_ratio):
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
     color_augmentation: different options regarding color augmentation
    Returns:
    A preprocessed image.
    """

    if ignore_aspect_ratio:
        # choose a wider range if apsect ratio is ignored
        resize_side = tf.random_uniform(
          [],
          minval=resize_side_min,
          maxval=int(1.2*resize_side_max)+1, dtype=tf.int32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
            image, size=[resize_side, resize_side])
        image = tf.squeeze(image)
        image.set_shape([None, None, 3])
    else:
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

    elif color_augmentation == 'little':
        fast_mode = True
        use_fast_color_distort = False

    elif color_augmentation == 'full_fast':
        fast_mode = True
        use_fast_color_distort = True

    elif color_augmentation == 'full_randomized':
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
                        resize_side,
                        ignore_aspect_ratio):
    """Preprocesses the given image for evaluation.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
    Returns:
    A preprocessed image.
    """
    if ignore_aspect_ratio:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
                    image,
                    size=[output_height, output_width])
        image = tf.squeeze(image)
        image.set_shape([None, None, 3])
    else:
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
                     color_augmentation=None,
                     ignore_aspect_ratio=False):
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
                                    color_augmentation,
                                    ignore_aspect_ratio)
    else:
        return preprocess_for_eval(image, output_height, output_width,
                                   image_means, image_stdevs,
                                   resize_side_min,
                                   ignore_aspect_ratio)


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
    hue_delta = 0.05
    upper_contrast = 1.3
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
        image = tf.image.random_hue(image, max_delta=hue_delta)
        image = tf.image.random_contrast(image, lower=0.9, upper=upper_contrast)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.9, upper=upper_contrast)
        image = tf.image.random_hue(image, max_delta=hue_delta)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.9, upper=upper_contrast)
        image = tf.image.random_hue(image, max_delta=hue_delta)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=hue_delta)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_contrast(image, lower=0.9, upper=upper_contrast)
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
