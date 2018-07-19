import tensorflow as tf
from models.resnet_keras_mod import ResnetBuilder
#from tensorflow.python.keras.layers import Input, MaxPooling2D, Conv2D, Flatten, Dropout, BatchNormalization, AveragePooling2D
#from tensorflow.python.keras.regularizers import l2
#from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
#from models.resnet_keras_mod import _handle_dim_ordering, _get_block, _conv_bn_relu, basic_block, _residual_block,  _bn_relu


def my_model_fn(features, labels, mode, params):

    K.set_session(tf.get_default_session())

    # Log current mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        K.set_learning_phase(0)
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))

    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
        K.set_learning_phase(0)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))
        K.set_learning_phase(1)

    # Create Model Architecture
    image_size_resnet = tuple(params["image_target_size"]) + \
                             (params["image_n_color_channels"],)

    inputs = Input(shape=image_size_resnet, tensor=features['images'])

    flatten1 = ResnetBuilder.build_resnet_18(
            inputs=inputs,
            num_outputs=params["n_classes"],
            output_names=params['output_labels'])

    logits = Dense(units=params["n_classes"][0], kernel_initializer="he_normal")(flatten1)

    predictions = {'props': tf.nn.softmax(logits, name='softmax_tensor'),
        'classes':tf.argmax(logits, axis=1)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels['labels/primary'])

    # Add weight decay to the loss.
    loss = cross_entropy + params['weight_decay'] * tf.losses.get_regularization_loss()

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # calculate learning rate
        learning_rate = tf.train.exponential_decay(
                                params['learning_rate'],
                                tf.train.get_global_step(),
                                100000, 0.96, staircase=True)

        optimizer = tf.train.MomentumOptimizer(
           #learning_rate=params['learning_rate'],
           learning_rate=learning_rate,
           momentum=params['momentum'])

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params['multi_gpu']:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = []
        for old_value, new_value in layer.updates:
            update_ops.append(tf.assign(old_value, new_value))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
     mode=mode,
     predictions=predictions,
     loss=loss,
     train_op=train_op)
