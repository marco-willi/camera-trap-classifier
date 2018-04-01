""" Model Library """
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import SGD, Adagrad, RMSprop
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, TensorBoard)
from tensorflow.python.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2)
from models.resnet_keras_mod import ResnetBuilder
from models.cats_vs_dogs import architecture_flat as cats_vs_dogs_arch


def create_model(model_name, input_feeder,
                 target_labels, n_classes_per_label_type):
    """ Returns specified model architecture """

    data = input_feeder()
    model_input = Input(tensor=data['images'])

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

    model = Model(inputs=model_input, outputs=all_target_outputs)

    target_tensors = {x: tf.cast(data[x], tf.float32)
                      for x in target_labels}

    opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
    # opt =  RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', 'sparse_top_k_categorical_accuracy'],
                  target_tensors=target_tensors)

    return model
