from tensorflow.python.keras._impl import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import layers


def architecture(inputs, num_classes):
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
    dense2 = Dense(num_classes, activation='softmax',name="labels/primary")(drop1)

    return dense2
