""" Prepare Model """
import logging

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2)
from tensorflow.python.keras.utils import multi_gpu_model

from models.resnet import ResnetBuilder
from models.small_cnn import architecture as small_cnn


def load_model_from_disk(path_to_model_on_disk):
    """ Load weights from disk and add to model """
    logging.info("Loading model from: %s" % path_to_model_on_disk)
    loaded_model = load_model(path_to_model_on_disk)
    return loaded_model


def get_non_output_layer_ids(model, label_indicator='label/'):
    """ get non-output layers of a model """
    layer_ids = [i for i, x in enumerate(model.layers)
                 if label_indicator not in x.name]
    return layer_ids


def copy_model_weights(from_model, to_model, incl_last=True):
    """ copy the model weights of one model to the other """
    if incl_last:
        to_model.set_weights(from_model.get_weights())
    else:
        # exclude output layers
        layer_ids_from = get_non_output_layer_ids(from_model)
        layer_ids_to = get_non_output_layer_ids(to_model)

        layers_to_copy_from = [from_model.layers[i] for i in layer_ids_from]
        layers_to_copy_to = [to_model.layers[i] for i in layer_ids_to]
        assert len(layers_to_copy_from) == len(layers_to_copy_to), \
            "Models dont match, cannot copy weights"

        for from_layer, to_layer in zip(layers_to_copy_from, layers_to_copy_to):
            logging.debug("Copy weights from: %s to %s" %
                          (from_layer.name, to_layer.name))
            to_layer.set_weights(from_layer.get_weights())


def set_specific_layers_to_random(model_trained, model_random, layer):
    """ Set all layers with and after layer_name to random """

    logging.info("Replacing layers of model with random layers")

    layer_names = [x.name for x in model_trained.layers]

    # check if target layer is in model
    if layer not in layer_names:
        logging.error("Layer %s not in model.layers" % layer)
        logging.error("Available Layers %s" % layer_names)
        raise IOError("Layer %s not in model.layers" % layer)

    # find layers which have to be kept unchanged
    id_to_set_random = layer_names.index(layer)

    # combine old, trained layers with new random layers
    comb_layers = model_trained.layers[0:id_to_set_random]
    new_layers = model_random.layers[id_to_set_random:]
    comb_layers.extend(new_layers)

    # define new model
    new_model = Sequential(comb_layers)

    # print layers of new model
    for layer, i in zip(new_model.layers, range(0, len(new_model.layers))):
        logging.debug("New model - layer %s: %s" % (i, layer.name))

    return new_model


def set_last_layer_to_random(model_trained, model_random):
    """ Set all layers with and after layer_name to random """

    logging.info("Replacing layers of model with random layers")

    layer_names = [x.name for x in model_trained.layers]
    layer = layer_names[-1]

    # find layers which have to be kept unchanged
    id_to_set_random = layer_names.index(layer)

    # combine old, trained layers with new random layers
    comb_layers = model_trained.layers[0:id_to_set_random]
    new_layers = model_random.layers[id_to_set_random:]
    comb_layers.extend(new_layers)

    # define new model
    new_model = Sequential(comb_layers)

    # print layers of new model
    for layer, i in zip(new_model.layers, range(0, len(new_model.layers))):
        logging.info("New model - layer %s: %s" % (i, layer.name))

    return new_model


def load_model_and_replace_output(model_old, model_new, new_output_layer):
    """ Load a model and replace the last layer """

    new_input = model_old.input

    # get old model output before last layer
    old_output = model_old.layers[-2].output

    # create a new output layer
    new_output = (new_output_layer)(old_output)

    # combine old model with new output layer
    new_model = Model(inputs=new_input,
                      outputs=new_output)

    logging.info("Replacing output layer of model")

    # print layers of old model
    for layer, i in zip(new_model.layers, range(0, len(new_model.layers))):
        logging.info("Old model - layer %s: %s" % (i, layer.name))

    return new_model


def set_last_layer_to_non_trainable(model):
    """ Set layers of a model to non-trainable """

    non_output_layers = get_non_output_layer_ids(model)

    layers_to_non_trainable = [model.layers[i] for i in non_output_layers]

    for layer in layers_to_non_trainable:
        layer.trainable = False

    for layer in model.layers:
        logging.debug("Layer %s is trainable: %s" %
                      (layer.name, layer.trainable))
    return model


def set_layers_to_non_trainable(model, layers):
    """ Set layers of a model to non-trainable """

    layers_to_non_trainable = [model.layers[i] for i in layers]

    for layer in layers_to_non_trainable:
        layer.trainable = False

    for layer in model.layers:
        logging.debug("Layer %s is trainable: %s" %
                      (layer.name, layer.trainable))
    return model


def create_model(model_name,
                 input_shape,
                 target_labels,
                 n_classes_per_label_type,
                 n_gpus,
                 train=True,
                 continue_training=False,
                 transfer_learning=False,
                 fine_tuning=False,
                 path_of_model_to_load=None,
                 initial_learning_rate=0.01
                 ):

    """ Returns specified model architecture """

    model_input = Input(shape=input_shape, name='images')

    if model_name == 'InceptionResNetV2':

        keras_model = InceptionResNetV2(
            include_top=False,
            weights=None,
            input_tensor=model_input,
            input_shape=None,
            pooling='avg'
        )

        output_flat = keras_model.output
        model_input = keras_model.input

    elif model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
                        'ResNet152']:
        res_builder = ResnetBuilder()
        if model_name == 'ResNet18':
            output_flat = res_builder.build_resnet_18(model_input)
        elif model_name == 'ResNet34':
            output_flat = res_builder.build_resnet_34(model_input)
        elif model_name == 'ResNet50':
            output_flat = res_builder.build_resnet_50(model_input)
        elif model_name == 'ResNet101':
            output_flat = res_builder.build_resnet_101(model_input)
        elif model_name == 'ResNet152':
            output_flat = res_builder.build_resnet_152(model_input)

    elif model_name == 'small_cnn':

        output_flat = small_cnn(model_input)
    else:
        raise ValueError("Model: %s not implemented" % model_name)

    all_target_outputs = list()

    for n_classes, target_name in zip(n_classes_per_label_type, target_labels):
        all_target_outputs.append(Dense(units=n_classes,
                                        kernel_initializer="he_normal",
                                        activation='softmax',
                                        name=target_name)(output_flat))

    # Define model optimizer
    opt = SGD(lr=initial_learning_rate, momentum=0.9, decay=1e-4)
    # opt = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

    model = Model(inputs=model_input, outputs=all_target_outputs)

    if continue_training:
        logging.debug("Preparing continue_training")
        loaded_model = load_model_from_disk(path_of_model_to_load)
        copy_model_weights(loaded_model, model, incl_last=True)

    elif transfer_learning:
        logging.debug("Preparing transfer_learning")
        loaded_model = load_model_from_disk(path_of_model_to_load)
        copy_model_weights(loaded_model, model, incl_last=False)
        non_output_layers = get_non_output_layer_ids(model)
        model = set_layers_to_non_trainable(model, non_output_layers)

    elif fine_tuning:
        logging.debug("Preparing fine_tuning")
        loaded_model = load_model_from_disk(path_of_model_to_load)
        copy_model_weights(loaded_model, model, incl_last=False)

    # Use multiple GPUs if available
    try:
        model = multi_gpu_model(model, n_gpus, cpu_relocation=True)
        logging.info("Training using multiple GPUs..")
    except:
        logging.info("Training using single GPU or CPU..")

    model.compile(loss='sparse_categorical_crossentropy',
                  #loss=sparse_loss,
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy',
                           'sparse_top_k_categorical_accuracy'])

    return model


def sparse_loss(y_true, y_pred):
    return tf.losses.sparse_softmax_cross_entropy(
        labels=tf.cast(y_true, tf.int64),
        logits=y_pred)
