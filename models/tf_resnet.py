import tensorflow as tf

from models.resnet_official.tf_resnet import (
    Model, building_block, bottleneck_block, _get_block_sizes)


def my_model_fn(features, labels, mode, params):
    """Shared functionality for different resnet model_fns.

    Initializes the ResnetModel representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.

    Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    multi_gpu: If True, wrap the optimizer in a TowerOptimizer suitable for
      data-parallel distribution across multiple GPUs.

    Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
    """

    #resnet_size = params['resnet_size']
    #multi_gpu = params['multi_gpu']
    resnet_size = 18
    multi_gpu = False

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
        block_fn = building_block
        final_size = 512
    else:
        block_fn = bottleneck_block
        final_size = 2048

    model = Model(
                resnet_size=resnet_size,
                num_filters=64,
                kernel_size=7,
                conv_stride=2,
                first_pool_size=3,
                first_pool_stride=2,
                second_pool_size=7,
                second_pool_stride=1,
                block_fn=block_fn,
                block_sizes=_get_block_sizes(resnet_size),
                block_strides=[1, 2, 2, 2],
                final_size=final_size)

    # Create Model Architecture
    inputs = features['images']

    flat_output = model(inputs, mode == tf.estimator.ModeKeys.TRAIN)

    head_list = list()
    for label, n_class in zip(params['output_labels'], params["n_classes"]):
        head_list.append(tf.contrib.learn.multi_class_head(
                            n_class,
                            loss_fn=tf.losses.sparse_softmax_cross_entropy,
                            label_name=label, head_name=label))

    head = tf.contrib.learn.multi_head(head_list)

    # 1. Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return head.create_model_fn_ops(
            features=features,
            mode=mode,
            logits=None,
            logits_input=flat_output
            ).estimator_spec()

    # 2. Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return head.create_model_fn_ops(
            features=features,
            mode=mode,
            labels=labels,
            logits=None,
            logits_input=flat_output
            ).estimator_spec()

    if mode == tf.estimator.ModeKeys.TRAIN:

        global_step = tf.train.get_or_create_global_step()

        # Define variable to optimize and use pattern matching if
        # transfer_learning is specified
        vars_to_optimize = tf.trainable_variables()
        if params['transfer_learning']:
            vars_to_optimize = [
                v for v in vars_to_optimize
                if params['transfer_learning_layers'] in v.name]

        learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        tf.summary.scalar('learning_rate_summary', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'])

        def loss_filter_fn(name):
            return 'batch_normalization' not in name

        def _train_op_fn(loss, optimizer=optimizer):
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
            #regularization = tf.losses.get_regularization_loss()

            loss_total = loss + params['weight_decay'] * tf.add_n(
              [tf.nn.l2_loss(v) for v in tf.trainable_variables()
               if loss_filter_fn(v.name)])
            return optimizer.minimize(loss_total, global_step)

        if multi_gpu:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        return head.create_model_fn_ops(
            features=features,
            mode=mode,
            labels=labels,
            train_op_fn=_train_op_fn,
            logits=None,
            logits_input=flat_output
            ).estimator_spec()
