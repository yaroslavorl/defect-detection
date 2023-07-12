from keras.models import Sequential
from keras.layers import (BatchNormalization, LeakyReLU,
                          MaxPooling3D, Conv3DTranspose, Conv3D)


def Net():

    model = Sequential()

    model.add(Conv3D(32, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling3D((2, 2, 2), (2, 2, 2)))

    model.add(Conv3D(48, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling3D((2, 2, 2), (2, 2, 2)))

    model.add(Conv3D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling3D((2, 2, 2), (2, 2, 2)))

    model.add(Conv3D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(48, 2, strides=(2, 2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3D(1, 3, activation='sigmoid', padding='same'))

    return model
