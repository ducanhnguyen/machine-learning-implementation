"""
# Batch normalization implementation (https://arxiv.org/abs/1502.03167)
# Data: digit-recognizer (number of classes = 10)
# Method: deep learning, relu activation on all layers except the last layer, softmax activation on the last layer,
# mini-batch gradient descent
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
        self.M1 = M1
        self.M2 = M2
        self.tf_W = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M1, M2), mean=0,
                                                                                 stddev=1))  # xavier intialization
        self.tf_b = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M2,)))

        # used when testing or validation (batch normalization)
        self.running_mean = tf.Variable(dtype=tf.float32, initial_value=np.zeros(M2))
        self.running_variance = tf.Variable(dtype=tf.float32, initial_value=np.zeros(M2))


class NeuralNetwork:

    def __init__(self, hiddenLayersSize):
        """
        :param hiddenLayersSize: 1-D dimension, represents the number of units in each hidden layer
        """
        self.hiddenLayersSize = hiddenLayersSize

    def initializeLayers(self, nFeatures, nClasses, hiddenLayersSize):
        """
        Initialize the input layer, hidden layers, and the output layer
        :param nFeatures: the number of features
        :param nClasses: the number of classes
        :param hiddenLayersSize: 1-D dimension, represents the number of units in each hidden layer
        :return: list of layers
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

    def fit(self, Xtrain, ytrain, Xval, yval, epoch=10, learning_rate=0.001, batch_size=50):
        """
        train model
        :param Xtrain: observations' input
        :param ytrain: observations' label
        :param epoch: the number of epoch for training
        :return:
        """
        self.decay = 0.9  # used to update the average mean and variance in batch normalization

        K = np.amax(ytrain) + 1  # get the number of classes
        Ytrain = utils.convert2indicator(ytrain)
        Yval = utils.convert2indicator(yval)
        N, D = Xtrain.shape

        layers = self.initializeLayers(nFeatures=D, nClasses=K, hiddenLayersSize=self.hiddenLayersSize)

        # initialize placeholders
        tf_X = tf.placeholder(dtype=tf.float32, name='X')
        tf_Y = tf.placeholder(dtype=tf.float32, name='Y')

        # define symbolic formula
        tf_Yhat_train = self.forward_train(tf_X, layers)
        tf_cost_train = tf.math.reduce_sum(-1 * tf.multiply(tf_Y, tf.log(tf_Yhat_train + 1e-4)))  # cross-entropy

        tf_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(tf_cost_train)

        tf_Yhat_test = self.forward_test(tf_X, layers)
        tf_cost_test = tf.math.reduce_sum(
            -1 * tf.multiply(tf_Y, tf.log(tf_Yhat_test + 1e-4)))  # cross-entropy, avoid NaN
        tf_yhat_test = tf.math.argmax(tf_Yhat_test, axis=1)

        # just for visualization
        trainingErrors = []
        validationErrors = []
        trainingAccuracies = []
        validationAccuracies = []

        iterations = []

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            iteration = 0
            nBatches = np.int(np.round(N * 1.0 / batch_size - 0.5))

            for i in range(epoch):

                for j in range(nBatches + 1):
                    print('iteration ' + str(iteration))
                    iterations.append(iteration)
                    iteration += 1

                    # mini-batch gradient descent
                    if j == nBatches:
                        session.run(tf_train, feed_dict={tf_X: Xtrain[j * nBatches:N],
                                                         tf_Y: Ytrain[j * nBatches:N]})
                    else:
                        session.run(tf_train, feed_dict={tf_X: Xtrain[j * nBatches:(j + 1) * nBatches],
                                                         tf_Y: Ytrain[j * nBatches:(j + 1) * nBatches]})

                    yhat_train = session.run(tf_yhat_test, feed_dict={tf_X: Xtrain, tf_Y: Ytrain})
                    accuracy = np.mean(yhat_train == ytrain)
                    print("training accuracy: " + str(accuracy))
                    trainingAccuracies.append(accuracy)

                    yhat_test = session.run(tf_yhat_test, feed_dict={tf_X: Xval, tf_Y: Yval})
                    accuracy = np.mean(yhat_test == yval)
                    print("validation accuracy: " + str(accuracy))
                    validationAccuracies.append(accuracy)

                    trainingError = session.run(tf_cost_test, feed_dict={tf_X: Xtrain, tf_Y: Ytrain}) / len(Xtrain)
                    print('training error: ' + str(trainingError))
                    trainingErrors.append(trainingError)

                    validationError = session.run(tf_cost_test, feed_dict={tf_X: Xval, tf_Y: Yval}) / len(Xval)
                    print('validation error: ' + str(validationError))
                    validationErrors.append(validationError)

                    print()

        self.plotError(trainingErrors, validationErrors, iterations)
        self.plotAccuracy(trainingAccuracies, validationAccuracies, iterations)

    def forward_train(self, tf_X, layers):
        """

        :param tf_X: observations' input
        :param layers: 1-D dimensions, includes input layer and hidden layers
        :param pkeep: 1-D dimensions
        :return:
        """
        tf_input = tf_X

        for idx, layer in enumerate(layers):
            # use tf.add to perform broadcasting
            tf_input = tf.add(x=tf.matmul(a=tf_input, b=layer.tf_W), y=layer.tf_b)

            if idx == len(layers) - 1:
                # do not apply batch normalization on the last layer
                tf_input = tf.nn.softmax(tf_input, axis=1)
            else:
                # batch normalization
                # compute mean and variance of each features with the given observations
                mean, variance = tf.nn.moments(x=tf_input, axes=[0])  # shape: (M2, 1)

                layer.running_mean = self.decay * layer.running_mean + (1 - self.decay) * mean  # shape: (M2, 1)
                layer.running_variance = self.decay * layer.running_variance + (
                        1 - self.decay) * variance  # shape: (M2, 1)

                # must perform batch normalization before using activation function
                tf_input = tf.nn.batch_normalization(
                    x=tf_input,
                    mean=mean,
                    variance=variance,
                    offset=None,  # no need for shift
                    scale=None,  # no need for scale
                    variance_epsilon=1e-4  # avoid the problem of division-by-zero in case that variance = 0
                )
                tf_input = tf.nn.relu(tf_input)

        return tf_input

    def forward_test(self, tf_X, layers):
        """
        Used in testing or validation
        :param tf_X: observations' input
        :param layers: 1-D dimensions, includes input layer and hidden layers
        :param pkeep: 1-D dimensions
        :return:
        """
        tf_input = tf_X

        for idx, layer in enumerate(layers):
            # use tf.add to perform broadcasting
            tf_input = tf.add(x=tf.matmul(a=tf_input, b=layer.tf_W), y=layer.tf_b)

            if idx == len(layers) - 1:
                # do not apply batch normalization on the last layer
                tf_input = tf.nn.softmax(tf_input, axis=1)
            else:
                tf_input = tf.nn.batch_normalization(
                    x=tf_input,
                    mean=layer.running_mean,  # the average of mean in the current layer
                    variance=layer.running_variance,  # the average of variance in the current layer
                    offset=None,
                    scale=None,
                    variance_epsilon=1e-4
                )
                tf_input = tf.nn.relu(tf_input)

        return tf_input

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
        plt.title('Batchnormalization (data = digit-recognizer)')
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
        plt.title('Batch normalization (data = digit-recognizer)')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    X, y = utils.readCsv('../data/digit-recognizer/train.csv', limit=2000)
    TRAIN = 1500
    Xtrain = X[:TRAIN]
    ytrain = y[:TRAIN]
    Xval = X[TRAIN:]
    yval = y[TRAIN:]

    s = NeuralNetwork(hiddenLayersSize=[64, 32])
    s.fit(Xtrain, ytrain, Xval, yval, epoch=30, learning_rate=0.01, batch_size=50)


if __name__ == "__main__":
    main()
