from tensorflow.python.keras._impl import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import layers


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
