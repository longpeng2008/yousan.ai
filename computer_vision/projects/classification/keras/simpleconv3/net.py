from keras.models import *
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense

def simpleconv3(input_shape=(48, 48, 3), classes=2):
    img_input = Input(shape=input_shape)

    bn_axis = 3
    x = Conv2D(12, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = Conv2D(24, (3, 3), strides=(2, 2), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x)
    x = Activation('relu')(x)

    x = Conv2D(48, (3, 3), strides=(2, 2), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv3')(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(img_input, x)
    return model
