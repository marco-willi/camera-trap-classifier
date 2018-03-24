import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input

from models.resnet_keras_mod import ResnetBuilder
from config.config import logging



def my_model_fn(features, labels, mode, params):

    K.set_session(tf.get_default_session())

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
            inputs=inputs)

    head_list = list()
    logits = dict()

    for label, n_class in zip(params['output_labels'], params["n_classes"]):
        if n_class > 2:
            current_head = tf.contrib.estimator.multi_class_head(
                            n_classes=n_class,
                            label_vocabulary=params['label_vocabulary'][label],
                            loss_reduction=tf.losses.Reduction.MEAN,
                            name=label)
        else:
            current_head = tf.contrib.estimator.binary_classification_head(
                label_vocabulary=params['label_vocabulary'][label],
                loss_reduction=tf.losses.Reduction.MEAN,
                name=label)
        head_list.append(current_head)
        logits_current = tf.layers.dense(inputs=flat_output, units=n_class)
        logits[label] = logits_current

    head = tf.contrib.estimator.multi_head(head_list)

    # 1. Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return head.create_estimator_spec(
            features=features,
            mode=mode,
            logits=logits
            )

    # 2. Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            logits=logits
            )

    if mode == tf.estimator.ModeKeys.TRAIN:

        global_step = tf.train.get_or_create_global_step()

        # Define variable to optimize and use pattern matching if
        # transfer_learning is specified
        vars_to_optimize = tf.trainable_variables()
        if params['transfer_learning']:
            vars_to_optimize = [v for v in vars_to_optimize
                                if params['transfer_learning_layers'] in v.name]

        learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        tf.summary.scalar('learning_rate_summary', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'])

        reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        def _train_op_fn(loss, reg_loss=reg_losses, optimizer=optimizer):
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
            # regularization = tf.losses.get_regularization_loss()
            # loss_total = loss + params['weight_decay'] * regularization
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
            #      if 'batch_normalization' not in v.name])
            loss_total = loss + reg_losses

            return optimizer.minimize(loss_total, global_step)
            #return train_op

        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            #optimizer=optimizer,
            train_op_fn=_train_op_fn,
            logits=logits
            #regularization_losses=reg_losses
            )
