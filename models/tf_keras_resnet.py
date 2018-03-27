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

    for label, n_class in zip(params['output_labels'], params["n_classes"]):
        if n_class > 2:
            metric_class_ids = [x for x in range(0, n_class)]
        else:
            metric_class_ids = None
        head_list.append(tf.contrib.learn.multi_class_head(
                            n_classes=n_class,
                            #loss_fn=tf.losses.sparse_softmax_cross_entropy,
                            metric_class_ids=metric_class_ids,
                            label_name=label, head_name=label))

    head = tf.contrib.learn.multi_head(head_list)
    #head = tf.contrib.estimator.multi_head(head_list)

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
            vars_to_optimize = [v for v in vars_to_optimize
                                if params['transfer_learning_layers'] in v.name]

        learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        #learning_rate = 1 / (1 + params['learning_rate_decay'] * global_step)

        #tf.summary.scalar('learning_rate_summary', learning_rate)

        # optimizer = tf.train.MomentumOptimizer(
        #     learning_rate=learning_rate,
        #     momentum=params['momentum'])

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            epsilon=0.1)

        def _train_op_fn(loss, optimizer=optimizer):
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)

            regularization = tf.losses.get_regularization_loss()
            loss_total = loss + regularization
            return optimizer.minimize(loss_total, global_step)

        return head.create_model_fn_ops(
            features=features,
            mode=mode,
            labels=labels,
            train_op_fn=_train_op_fn,
            logits=None,
            logits_input=flat_output
            ).estimator_spec()
