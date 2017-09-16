#!/usr/bin/env python

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils import plot_model


def make_correspondence_model(left, right):
    x = Concatenate()([left, right])
    x = Conv2D(32, (5, 5))(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    def m(xin):
        xin = Conv2D(32, (5, 5), strides=(4, 4))(xin)
        xin = Dropout(0.2)(xin)
        return xin

    conv = [m(x) for i in range(4)]

    x = Concatenate()(conv)

    return Model(inputs=(left, right), outputs=x)


def make_model():
    left = Input(shape=(540, 960, 3))
    right = Input(shape=(540, 960, 3))
    m = make_correspondence_model(left, right)
    plot_model(m, to_file='/tmp/a.png', show_shapes=True)

