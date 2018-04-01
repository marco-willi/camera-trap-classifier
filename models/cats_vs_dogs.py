from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops


def architecture(inputs, num_classes, output_names):
    """ Architecture of model """

    # Check input types
    assert (isinstance(num_classes, int) or isinstance(num_classes, list)) and \
           (isinstance(output_names, str) or isinstance(output_names, list))

    # convert to list
    if isinstance(num_classes, int):
        num_classes = [num_classes]

    if isinstance(output_names, str):
        output_names = [output_names]

    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(max1)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(max2)
    max3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat1 = Flatten()(max3)
    dense1 = Dense(64, activation='relu')(flat1)
    drop1 = Dropout(0.5)(dense1)

    # create multiple output
    all_outputs = list()
    for n, name in zip(num_classes, output_names):
        all_outputs.append(Dense(n, activation='softmax', name=name)(drop1))

    return all_outputs


def architecture_flat(inputs):
    """ Architecture of model """

    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(max1)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(max2)
    max3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat1 = Flatten()(max3)
    dense1 = Dense(64, activation='relu')(flat1)
    drop1 = Dropout(0.5)(dense1)

    return drop1




def architecture_estimator(inputs, num_classes, output_names):
    """ Architecture of model """

    # Check input types
    assert (isinstance(num_classes, int) or isinstance(num_classes, list)) and \
           (isinstance(output_names, str) or isinstance(output_names, list))

    # convert to list
    if isinstance(num_classes, int):
        num_classes = [num_classes]

    if isinstance(output_names, str):
        output_names = [output_names]

    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(max1)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(max2)
    max3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat1 = Flatten()(max3)
    dense1 = Dense(64, activation='relu')(flat1)
    drop1 = Dropout(0.5)(dense1)

    # create multiple output
    all_outputs = list()
    for n, name in zip(num_classes, output_names):
        all_outputs.append(Dense(n, activation='softmax', name=name)(drop1))

    return all_outputs


def architecture_mf(inputs):
    """ Architecture of model """

    conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(max1)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(max2)
    max3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat1 = Flatten()(max3)
    dense1 = Dense(64, activation='relu')(flat1)
    drop1 = Dropout(0.5)(dense1)

    return drop1


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

    flat_output = architecture_mf(inputs)

    def _calc_loss(logits, labels, weights=1.0):
        two_class_logits = array_ops.concat((array_ops.zeros_like(logits), logits), 1)
        return tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=two_class_logits, weights=weights)

    def _calc_loss2(logits, labels, weights=1.0):
        return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=weights)

    def _calc_loss3(logits, labels, weights=1.0):
        labels2 = tf.to_float(labels)
        return tf.contrib.keras.backend.binary_crossentropy(output=logits, target=labels2, from_logits=True)

    head_list = list()
    for label, n_class in zip(params['output_labels'], params["n_classes"]):
        head_list.append(tf.contrib.learn.multi_class_head(
                            n_class,
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
            vars_to_optimize = [v for v in vars_to_optimize
                                if params['transfer_learning_layers'] in v.name]

        learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        tf.summary.scalar('learning_rate_summary', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'])

        def _train_op_fn(loss, optimizer=optimizer):
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)

            regularization = tf.losses.get_regularization_loss()
            loss_total = loss + params['weight_decay'] * regularization
            return optimizer.minimize(loss_total, global_step)

        return head.create_model_fn_ops(
            features=features,
            mode=mode,
            labels=labels,
            train_op_fn=_train_op_fn,
            logits=None,
            logits_input=flat_output
            ).estimator_spec()
