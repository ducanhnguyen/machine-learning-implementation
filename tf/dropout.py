"""
# Data: digit-recognizer (number of classes = 10)
# Method: deep learning, softmax activation on all layers, stochatic gradient descent
# Dropout implementation: only apply dropout when training, no dropout when testing
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

import utils


class Layer:
    """
    Represent a layer in neural network, i.e., input layer or hidden layer
    """

    def __init__(self, M1, M2):
        """
        Initialize a layer
        :param M1: The number of hidden units in the current layer
        :param M2: the number of hidden units in the next layer
        """
        self.tf_W = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M1, M2), mean=0, stddev=1))
        self.tf_b = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M2,)))


class NeuralNetwork:
    def __init__(self, hiddenLayersSize, pkeep):
        """

        :param hiddenLayersSize: 1-D dimension, represents the number of units in each hidden layer
        :param pkeep: 1-D dimentions, used for droput
        """
        self.hiddenLayersSize = hiddenLayersSize
        self.pkeep = pkeep

    def initializeLayers(self, nFeatures, nClasses, hiddenLayersSize):
        """
        Initialize the input layer and hidden layers
        :param nFeatures: the number of features
        :param nClasses: the number of classes
        :param hiddenLayersSize: 1-D dimension, represents the number of units in each hidden layer
        :return:
        """
        layers = []
        inputLayer = Layer(nFeatures, hiddenLayersSize[0])
        layers.append(inputLayer)

        for idx, numUnits in enumerate(hiddenLayersSize):

            if idx == len(hiddenLayersSize) - 1:
                hiddenLayer = Layer(numUnits, nClasses)
                layers.append(hiddenLayer)
            else:
                hiddenLayer = Layer(numUnits, hiddenLayersSize[idx + 1])
                layers.append(hiddenLayer)
        return layers

    def fit(self, Xtrain, ytrain, Xval, yval, epoch=1, learning_rate=0.01):
        """
        train model
        :param Xtrain: observations' input
        :param ytrain: observations' label
        :param epoch: the number of epoch for training
        :return:
        """
        K = np.amax(ytrain) + 1  # get the number of classes
        Ytrain = utils.convert2indicator(ytrain)
        Yval = utils.convert2indicator(yval)
        N, D = Xtrain.shape

        layers = self.initializeLayers(nFeatures=D, nClasses=K, hiddenLayersSize=self.hiddenLayersSize)

        # initialize placeholders
        tf_X = tf.placeholder(dtype=tf.float32, name='X')
        tf_Y = tf.placeholder(dtype=tf.float32, name='Y')

        # define symbolic formula
        tf_Yhat_training = self.forward_training(tf_X, layers, self.pkeep)  # backpropogation during training
        tf_cost_training = tf.math.reduce_sum(-1 * tf.multiply(tf_Y, tf.log(tf_Yhat_training)))  # cross-entropy

        tf_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(tf_cost_training)

        # we do not use dropout when testing
        tf_Yhat_testing = self.forward_test(tf_X, layers)  # backpropogation during testing
        tf_cost_testing = tf.math.reduce_sum(-1 * tf.multiply(tf_Y, tf.log(tf_Yhat_testing)))  # cross-entropy

        tf_yhat = tf.math.argmax(tf_Yhat_training, axis=1)

        trainingErrors = []
        validationErrors = []
        trainingAccuracies = []
        validationAccuracies = []
        iterations = []

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            iteration = 0
            for i in range(epoch):

                for j in range(len(Xtrain)):
                    print('iteration ' + str(iteration))
                    iterations.append(iteration)
                    iteration += 1

                    # stochatic gradient descent
                    session.run(tf_train, feed_dict={tf_X: Xtrain[j:j+1], tf_Y: Ytrain[j:j+1]})

                    yhat = session.run(tf_yhat, feed_dict={tf_X: Xtrain, tf_Y: Ytrain})
                    accuracy = np.mean(yhat == ytrain)
                    print("training accuracy: " + str(accuracy))
                    trainingAccuracies.append(accuracy)

                    yhat = session.run(tf_yhat, feed_dict={tf_X: Xval, tf_Y: Yval})
                    accuracy = np.mean(yhat == yval)
                    print("validation accuracy: " + str(accuracy))
                    validationAccuracies.append(accuracy)

                    trainingError = session.run(tf_cost_testing, feed_dict={tf_X: Xtrain, tf_Y: Ytrain}) / len(Xtrain)
                    print('training error: ' + str(trainingError))
                    trainingErrors.append(trainingError)

                    validationError = session.run(tf_cost_testing, feed_dict={tf_X: Xval, tf_Y: Yval}) / len(Xval)
                    print('validation error: ' + str(validationError))
                    validationErrors.append(validationError)

                    print()

        self.plotError(trainingErrors, validationErrors, iterations)
        self.plotAccuracy(trainingAccuracies, validationAccuracies, iterations)

    def forward_training(self, tf_X, layers, pkeep):
        """

        :param tf_X: observations' input
        :param layers: 1-D dimensions, includes input layer and hidden layers
        :param pkeep: 1-D dimensions
        :return:
        """
        tf_input = tf_X

        for idx, layer in enumerate(layers):
            tf_input = tf.nn.dropout(x=tf_input, keep_prob=pkeep[idx])  # dropout
            tf_input = tf.nn.softmax(tf.matmul(tf_input, layer.tf_W) + layer.tf_b)

        return tf_input

    def forward_test(self, tf_X, layers):
        """

        :param tf_X: observations' input
        :param layers: 1-D dimensions, includes input layer and hidden layers
        :return:
        """
        tf_input = tf_X

        for idx, layer in enumerate(layers):
            tf_input = tf.nn.softmax(tf.matmul(tf_input, layer.tf_W) + layer.tf_b)

        return tf_input

    def predict(self, X, y):
        yhat = self.forward_training(X, y)
        return np.mean(yhat == y)

    def plotError(self, trainingErrors, validationErrors, iterations):
        """
        Visualization
        :param scores: 1-D dimension of float numbers
        :param iterations:  1-D dimension of integer numbers
        :return:
        """
        plt.plot(iterations, validationErrors, label="validation error")
        plt.plot(iterations, trainingErrors, label="training error")
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Dropout (data = digit-recognizer)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plotAccuracy(self, trainingAccuracy, validationAccuracy, iterations):
        """
        Visualization
        :param scores: 1-D dimension of float numbers
        :param iterations:  1-D dimension of integer numbers
        :return:
        """
        plt.plot(iterations, validationAccuracy, label="validation accuracy")
        plt.plot(iterations, trainingAccuracy, label="training accuracy")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Dropout (data = digit-recognizer)')
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    X, y = utils.readCsv('../data/digit-recognizer/train.csv', limit=5000)
    TRAIN = 4000
    Xtrain = X[:TRAIN]
    ytrain = y[:TRAIN]
    Xval = X[TRAIN:]
    yval = y[TRAIN:]

    #s = NeuralNetwork(hiddenLayersSize=[10, 5, 5], pkeep=[0.8, 0.5, 0.5, 1])
    s = NeuralNetwork(hiddenLayersSize=[10, 5, 5], pkeep=[1, 1, 1, 1])
    s.fit(Xtrain, ytrain, Xval, yval)


if __name__ == "__main__":
    main()
