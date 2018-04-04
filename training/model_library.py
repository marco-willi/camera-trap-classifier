""" Model Library """
import logging

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import SGD, Adagrad, RMSprop
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, TensorBoard)
from tensorflow.python.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2)
from tensorflow.python.keras.utils import multi_gpu_model

from models.resnet_keras_mod import ResnetBuilder
from models.cats_vs_dogs import architecture_flat as cats_vs_dogs_arch


def load_model_from_disk(path_to_model_on_disk):
    """ Load weights from disk and add to model """
    logging.info("Loading model from: %s" % path_to_model_on_disk)
    loaded_model = load_model(path_to_model_on_disk)
    return loaded_model


def copy_model_weights(from_model, to_model, incl_last=True):
    """ copy the model weights of one model to the other """
    if incl_last:
        to_model.set_weights(from_model.get_weights())
    else:
        layers_to_copy_from = from_model.layers[0:-1]
        layers_to_copy_to = to_model.layers[0:-1]
        assert len(layers_to_copy_from) == len(layers_to_copy_to), \
            "Models dont match, cannot copy weights"

        for from_layer, to_layer in zip(layers_to_copy_from, layers_to_copy_to):
            logging.info("Copy weights from: %s to %s" % (from_layer.name, to_layer.name))
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
        logging.info("New model - layer %s: %s" % (i, layer.name))

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


def set_layers_to_non_trainable(model, first_layer_to_train):
    """ Set layers of a model to non-trainable """

    layer_names = [x.name for x in model.layers]

    # check if layer name is in model
    if first_layer_to_train not in layer_names:
        logging.error("Layer %s not in model.layers" %
                      first_layer_to_train)
        logging.error("Available Layers %s" %
                      layer_names)
        raise IOError("Layer %s not in model.layers" %
                      first_layer_to_train)

    # look for specified layer and set all previous layers
    # to non-trainable
    n_retrain = layer_names.index(first_layer_to_train)
    for layer in model.layers[0:n_retrain]:
        layer.trainable = False

    logging.info("Setting layers before %s to non-trainable" %
                 first_layer_to_train)

    for layer in model.layers:
        logging.info("Layer %s is trainable: %s" %
                     (layer.name, layer.trainable))
    return model


def set_last_layer_to_non_trainable(model):
    """ Set layers of a model to non-trainable """

    layer_names = [x.name for x in model.layers]
    first_layer_to_train = layer_names[-1]

    # look for specified layer and set all previous layers
    # to non-trainable
    n_retrain = layer_names.index(first_layer_to_train)
    for layer in model.layers[0:n_retrain]:
        layer.trainable = False

    logging.info("Setting layers before %s to non-trainable" %
                 first_layer_to_train)

    for layer in model.layers:
        logging.info("Layer %s is trainable: %s" %
                     (layer.name, layer.trainable))
    return model


def create_model(model_name,
                 target_labels, n_classes_per_label_type,
                 input_feeder=None, n_gpus=1,
                 train=True, test_input_shape=None,
                 continue_training=False,
                 transfer_learning=False,
                 path_of_model_to_load=None,
                 ):

    """ Returns specified model architecture """

    if train:
        assert input_feeder is not None, \
            "input_feeder must be specified with train=True (default)"
        data = input_feeder()
        model_input = Input(tensor=data['images'])
    else:
        model_input = Input(shape=test_input_shape)

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

    elif model_name == 'cats_vs_dogs':

        output_flat = cats_vs_dogs_arch(model_input)
    else:
        raise ValueError("Model: %s not implemented" % model_name)

    all_target_outputs = list()

    for n_classes, target_name in zip(n_classes_per_label_type, target_labels):
        all_target_outputs.append(Dense(units=n_classes,
                                        kernel_initializer="he_normal",
                                        activation='softmax',
                                        name=target_name)(output_flat))

    if not train:
        return Model(inputs=model_input, outputs=all_target_outputs)

    target_tensors = {x: tf.cast(data[x], tf.float32)
                      for x in target_labels}

    opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
    # opt =  RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

    if n_gpus > 1:
        with tf.device('/cpu:0'):
            base_model = Model(inputs=model_input, outputs=all_target_outputs)

            if continue_training:
                loaded_model = load_model_from_disk(path_of_model_to_load)
                copy_model_weights(loaded_model, base_model, incl_last=True)

            elif transfer_learning:
                loaded_model = load_model_from_disk(path_of_model_to_load)
                copy_model_weights(loaded_model, base_model, incl_last=False)
                #random_model = Model(inputs=model_input, outputs=all_target_outputs)
                #base_model = set_last_layer_to_random(base_model, random_model)
                base_model = set_last_layer_to_non_trainable(base_model)

        model = multi_gpu_model(base_model, gpus=n_gpus)

    else:
        model = Model(inputs=model_input, outputs=all_target_outputs)

        if continue_training:
            loaded_model = load_model_from_disk(path_of_model_to_load)
            copy_model_weights(loaded_model, model, incl_last=True)

        elif transfer_learning:
            loaded_model = load_model_from_disk(path_of_model_to_load)
            copy_model_weights(loaded_model, model, incl_last=False)
            #random_model = Model(inputs=model_input, outputs=all_target_outputs)
            #model = set_last_layer_to_random(model, random_model)
            model = set_last_layer_to_non_trainable(model)

        base_model = model

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', 'sparse_top_k_categorical_accuracy'],
                  target_tensors=target_tensors)

    return model, base_model
