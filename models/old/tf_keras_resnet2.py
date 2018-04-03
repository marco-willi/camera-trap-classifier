import tensorflow as tf
from models.resnet_keras_mod import ResnetBuilder
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from config.config import logging
from tensorflow.python.keras import backend as K


def my_model_fn(features, labels, mode, params):

    K.set_session(tf.get_default_session())
    # if tf.add_n(tf.is_nan(labels['labels/primary'])) > 0:
    #     logging.debug("NAN Loss")


    # Log current mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
        K.set_learning_phase(0)

    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
        K.set_learning_phase(0)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))
        K.set_learning_phase(1)

    # randomly save images for Tensorboard
    tf.summary.image('images', features['images'], max_outputs=6)

    # Create Model Architecture
    image_size_resnet = tuple(params["image_target_size"]) + \
                             (params["image_n_color_channels"],)

    inputs = Input(shape=image_size_resnet, tensor=features['images'])

    flat_output = ResnetBuilder.build_resnet_18(
            inputs=inputs,
            num_outputs=params["n_classes"],
            output_names=params['output_labels'])

    predictions = dict()
    logits = dict()
    for n_cl, label in zip(params["n_classes"], params['output_labels']):
        logits[label] = Dense(units=n_cl, kernel_initializer="he_normal")(flat_output)
        predictions[label + '_props'] = tf.nn.softmax(logits[label], name='softmax_tensor')
        predictions[label + '_classes'] = tf.argmax(logits[label], axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    for i, label in enumerate(params['output_labels']):
        if i == 0:
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                logits=logits[label], labels=labels[label])
        else:
            cross_entropy = tf.add(tf.losses.sparse_softmax_cross_entropy(
                logits=logits[label], labels=labels[label]),
                cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy #+ params['weight_decay'] * tf.losses.get_regularization_loss()
    logging.info("Loss: %s" % loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # calculate learning rate
        learning_rate = tf.train.exponential_decay(
                                params['learning_rate'],
                                tf.train.get_global_step(),
                                100000, 0.96, staircase=True)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
           learning_rate=learning_rate,
           momentum=params['momentum'])

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params['multi_gpu']:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
     mode=mode,
     predictions=predictions,
     loss=loss,
     train_op=train_op)
