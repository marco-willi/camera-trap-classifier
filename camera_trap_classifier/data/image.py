""" Functions to handle / process images """
import math

import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops

# CONSTANTS
CB_DISTORTION_RANGE = 0.05
CR_DISTORTION_RANGE = 0.05


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


def gaussian_kernel_2D(sigma=0.84, kernel_radius=None):
    """ Guassian (square) 2D Kernel with radius kernel_radius from center
        Default kernel_radius is 3 * sigma:
           https://en.wikipedia.org/wiki/Gaussian_blur
    """

    sigma = tf.cast(sigma, tf.float32)
    normal_dist = tf.distributions.Normal(loc=tf.cast(0, tf.float32),
                                          scale=sigma)

    # points in distance > (3 * sigma) can be safely ignored
    if kernel_radius is None:
        kernel_radius = tf.ceil(3 * sigma)

    probs = normal_dist.prob(tf.range(-kernel_radius, kernel_radius + 1, 1,
                                      dtype=tf.float32))

    n_vals = tf.cast(probs.shape[0], tf.int32)

    kernel_vals = tf.tile(probs, tf.expand_dims(n_vals, -1))

    kernel_vals_2D = tf.reshape(kernel_vals, shape=(n_vals, n_vals))

    kernel = tf.multiply(kernel_vals_2D, tf.transpose(kernel_vals_2D))

    normalized_kernel = kernel / tf.reduce_sum(kernel)

    return normalized_kernel


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
    """ Aspect Preserving image resizing
        The smaller side of the image will be of size smallest_side
    """
    # Calculate aspect ratio preserving heights / widths
    input_shape = tf.shape(image)
    input_height = input_shape[0]
    input_width = input_shape[1]
    new_height, new_width = \
        _smallest_size_at_least(input_height, input_width, smallest_side)
    image = tf.expand_dims(image, 0)
    # Resize the image while preserving the aspect ratio
    image = tf.image.resize_bilinear(
                image,
                size=[new_height, new_width])
    image = tf.squeeze(image, 0)
    return image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         image_means,
                         image_stdevs,
                         color_augmentation,
                         preserve_aspect_ratio,
                         zoom_factor,
                         crop_factor,
                         rotate_by_angle,
                         randomly_flip_horizontally):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: Deprecated
    resize_side_max: Deprecated
    zoom_factor: factor (0-0.5) to randomly zoom in or out of the image
                  0 is no zooming, 0.5 is zoom in or out between 50%-150%
    crop_factor: factor (0-0.5) to randomly crop a part of the image,
                 0 is no cropping, 0.5 is up to 50% cropping along both image
                 dimensions
    rotate_by_angle: randomly rotate image by plus/minus [0, angle] (degree)
    Returns:
    A preprocessed image.
    """
    # check inputs
    if zoom_factor < 0.0 or zoom_factor > 0.5:
        raise ValueError('zoom_factor  must be within [0, 0.5]')

    if crop_factor < 0.0 or crop_factor > 0.5:
        raise ValueError('crop_factor  must be within [0, 0.5]')

    # randomly zoom image
    if zoom_factor > 0:
        input_shape = tf.shape(image)
        input_height = tf.to_float(input_shape[0])
        input_width = tf.to_float(input_shape[1])
        zoom_proportion = tf.random_uniform(
            [], minval=1-zoom_factor, maxval=1+zoom_factor, dtype=tf.float32)
        height_zoom = tf.to_int32(tf.rint(zoom_proportion * input_height))
        width_zoom = tf.to_int32(tf.rint(zoom_proportion * input_width))
        image = tf.image.resize_image_with_crop_or_pad(
            image,
            height_zoom,
            width_zoom)

    # Randomly crop image
    if crop_factor > 0:
        input_shape = tf.shape(image)
        input_height = tf.to_float(input_shape[0])
        input_width = tf.to_float(input_shape[1])
        crop_proportion = tf.random_uniform(
            [], minval=1-crop_factor, maxval=1, dtype=tf.float32)
        height_crop = tf.to_int32(
            tf.rint(crop_proportion * tf.to_float(input_height)))
        width_crop = tf.to_int32(
            tf.rint(crop_proportion * tf.to_float(input_width)))
        image = _random_crop([image], height_crop, width_crop)[0]

    # Randomly rotate image
    if rotate_by_angle > 0:
        angle_radians = (rotate_by_angle * tf.constant(math.pi)) / 180
        angle_rand = tf.random_uniform(
            [], minval=-angle_radians, maxval=angle_radians)
        image = tf.expand_dims(image, 0)
        image = tf.contrib.image.rotate(image, angle_rand)
        image = tf.squeeze(image, 0)

    # Resize image to target width
    if not preserve_aspect_ratio:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
                    image,
                    size=[output_height, output_width])
        image = tf.squeeze(image, 0)
    else:
        smallest_side = tf.minimum(output_height, output_width)
        image = _aspect_preserving_resize(image, smallest_side)
        # Crop image centrally to the target output width / height
        image = _central_crop([image], output_height, output_width)[0]

    image.set_shape([output_height,  output_width, 3])
    image = tf.cast(image, tf.float32)

    # randomly flip image
    if randomly_flip_horizontally:
        image = tf.image.random_flip_left_right(image)

    # Convert image to range 0 and 1
    image = tf.divide(image, tf.cast(255.0, tf.float32))

    # Color Augmentation
    if color_augmentation is None:
        image = _image_standardize(image, image_means, image_stdevs)
        return image

    if color_augmentation == 'little':
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
                        preserve_aspect_ratio):
    """Preprocesses the given image for evaluation.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
    Returns:
    A preprocessed image.
    """
    # Directly resize the image to the target size if we ignore the
    # aspect ratio of the input image
    if not preserve_aspect_ratio:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
                    image,
                    size=[output_height, output_width])
        image = tf.squeeze(image, 0)
    else:
        smallest_side = tf.minimum(output_height, output_width)
        image = _aspect_preserving_resize(image, smallest_side)
        # Crop image centrally to the target output width / height
        image = _central_crop([image], output_height, output_width)[0]
    # standardize image
    image.set_shape([output_height, output_width, 3])
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, tf.cast(255.0, tf.float32))
    image = _image_standardize(image, image_means, image_stdevs)

    return image


def decode_image_bytes_1D(image_bytes_list,
                          output_height=None,
                          output_width=None,
                          image_choice_for_sets='random',
                          **kwargs):
    """ Decode a 1D Tensor of 1-N raw image bytes
    Args:
    image_bytes_list: a 1-D tensor with raw bytes
    output_height: height in pixels of decoded images
        (only used if image_choice_for_sets is not random)
    output_width: height in pixels of decoded images
        (only used if image_choice_for_sets is not random)
    """

    if image_choice_for_sets == 'random':
        image = choose_random_image(image_bytes_list)
    elif image_choice_for_sets == 'grayscale_stacking':
        image = grayscale_stacking_and_blurring(
                    image_bytes_list,
                    output_height, output_width)
    else:
        raise NotImplemented("Image choice for set: %s not implemented" %
                             image_choice_for_sets)
    return image


def preprocess_image(image, output_height, output_width,
                     is_training,
                     zoom_factor=0,
                     crop_factor=0,
                     rotate_by_angle=0,
                     image_means=[0, 0, 0],
                     image_stdevs=[1, 1, 1],
                     color_augmentation=None,
                     preserve_aspect_ratio=False,
                     randomly_flip_horizontally=True,
                     **kwargs):
    """Preprocesses the given image.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    Returns:
    A preprocessed image.
    """

    if is_training:
        return preprocess_for_train(image=image,
                                    output_height=output_height,
                                    output_width=output_width,
                                    image_means=image_means,
                                    image_stdevs=image_stdevs,
                                    color_augmentation=color_augmentation,
                                    preserve_aspect_ratio=preserve_aspect_ratio,
                                    zoom_factor=zoom_factor,
                                    crop_factor=crop_factor,
                                    rotate_by_angle=rotate_by_angle,
                                    randomly_flip_horizontally=randomly_flip_horizontally)
    else:
        return preprocess_for_eval(image=image,
                                   output_height=output_height,
                                   output_width=output_width,
                                   image_means=image_means,
                                   image_stdevs=image_stdevs,
                                   preserve_aspect_ratio=preserve_aspect_ratio)


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
        [], -CB_DISTORTION_RANGE, CB_DISTORTION_RANGE, seed=None)
    cr_factor = random_ops.random_uniform(
        [], -CR_DISTORTION_RANGE, CR_DISTORTION_RANGE, seed=None)

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

    return [tf.image.crop_to_bounding_box(
            image,
            offset_height,
            offset_width,
            crop_height,
            crop_width
            ) for image in image_list]


def choose_random_image(image_bytes_list):
    """ Choose a random image """
    n_images = tf.shape(image_bytes_list)

    # select a random image of the record
    rand = tf.random_uniform([], minval=0, maxval=n_images[0],
                             dtype=tf.int32)

    # decode image to tensor
    image = tf.image.decode_jpeg(image_bytes_list[rand])

    return image


def _decode_image_bytes_example(
        image_bytes,
        output_height=None, output_width=None, n_colors=3):
    """ Input is one TFRecord Exaample
        Example with three images:
            TensorShape([Dimension(1), Dimension(3)])
        Example Output:
            TensorShape([Dimension(3), Dimension(375),
                         Dimension(500), Dimension(3)])
    """
    if (output_width is None) or (output_height is None):
        images = tf.map_fn(
                    lambda x: tf.image.decode_jpeg(x, channels=n_colors),
                    image_bytes, dtype=tf.uint8)
        images = tf.cast(images, tf.float32)
    else:
        images = tf.map_fn(
                     lambda x: tf.image.resize_images(
                                 tf.image.decode_jpeg(x, channels=n_colors),
                                 [output_height, output_width]),
                     image_bytes, dtype=tf.float32)
    return images


def _stack_images_to_3D(image_tensor):
    """ Stack images """
    input_shape = image_tensor.get_shape().as_list()
    if input_shape[-1] == 1:
        target_shape = image_tensor.get_shape().as_list()
        target_shape[-1] = 3
        image_tensor = tf.broadcast_to(image_tensor, target_shape)
    elif input_shape[-1] == 2:
        image_tensor = tf.stack([
            image_tensor[:, :, 0],
            image_tensor[:, :, 1],
            image_tensor[:, :, 1]], 2)
    image_tensor.set_shape([input_shape[0], input_shape[1], 3])
    return image_tensor


def _blurr_imgs(img_batch):
    """ Blurr image batch with Gaussian Filter """
    with tf.variable_scope("gauss_kernel"):
        gauss_kernel = gaussian_kernel_2D(sigma=2)
        gauss_kernel = tf.expand_dims(tf.expand_dims(gauss_kernel, -1), -1)

    img_batch_blurred = tf.nn.conv2d(
        img_batch,
        filter=gauss_kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
        use_cudnn_on_gpu=True,
        data_format='NHWC'
    )

    return img_batch_blurred


def grayscale_stacking_and_blurring(
        image_bytes, output_height=None, output_width=None):
    """ Get and convert all images to grayscale """

    # Grayscale image batch tensor (4-D, NHWC)
    imgs = _decode_image_bytes_example(
        image_bytes, output_height, output_width, n_colors=1)

    # Apply Gaussian Blurring
    # Batch of 1-N blurred images
    imgs_blurred = _blurr_imgs(imgs)

    # Stack into RGB image, handle cases when there is only 1 or 2 images
    image = tf.transpose(tf.squeeze(imgs_blurred, -1), perm=[1, 2, 0])
    image = _stack_images_to_3D(image)

    image = tf.cast(image, tf.uint8)

    return image
