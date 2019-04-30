"""
Example of Sequential model in Keras
"""
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

import utils


def build_model(Xtrain, Ytrain):
    N, D = Xtrain.shape
    model = Sequential()

    model.add(
        # output = activation(dot(input, kernel) + bias)
        Dense(
            input_dim=D
            , units=64
            , activation='tanh'  # tanh, linear, softmax, softplus, softsign, sigmoid, elu, relu, selu, etc.

            , use_bias=True
            , bias_initializer=keras.initializers.Zeros()  # zeros, RandomNormal, RandomUniform, he_normal, etc.
            , bias_regularizer=keras.regularizers.l2(l=0.)  # l1, l2

            , kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=np.math.sqrt(1.0 / D))
            , kernel_regularizer=keras.regularizers.l2(l=0.)  # l1, l2
        )
    )

    model.add(Dense(units=10, activation='softmax'))

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    history = model.fit(x=Xtrain, y=Ytrain, batch_size=100, epochs=200, shuffle=True,
                        validation_split=0.2  # only train on 80%
                        )

    # plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def main():
    Xtrain, ytrain = utils.readTrainingDigitRecognizer('../data/digit-recognizer/train.csv')
    Ytrain = utils.convert2indicator(ytrain)
    build_model(Xtrain, Ytrain)


main()
