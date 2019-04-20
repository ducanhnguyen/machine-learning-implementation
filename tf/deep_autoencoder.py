"""
Implementation of AutoEncoder with more than hidden layer.

Kaggle: ~95%
"""
import numpy as np
import tensorflow as tf

import utils


class OutputLayer:
    def __init__(self, M, K):
        """
        :param K: Number of classes
        :param M: Number of units in the hidden layer
        """
        self.M = M
        self.K = K

        self.tf_W = tf.Variable(dtype=tf.float64,
                                initial_value=tf.random.normal(dtype=tf.float64, shape=(self.M, self.K), mean=0,
                                                               stddev=1 / self.M))
        self.tf_b = tf.Variable(dtype=tf.float64, initial_value=tf.random.normal(dtype=tf.float64, shape=(self.K,)))

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

    def forwardTraining(self, X):
        return tf.nn.softmax(tf.math.add(tf.matmul(a=X, b=self.tf_W), self.tf_b))


class AutoEncoder:
    """
    Represent a simple autoencoder having 1 hidden layer
    """

    def __init__(self, M1, M2):
        """

        :param M1: the nummber of units in input layer
        :param M2: the number of units in hidden layer
        """
        self.M2 = M2
        self.M1 = M1

    def fit(self, X, learning_rate=0.001, epoch=20, batch_size=100):
        N, _ = X.shape
        tf_X = tf.placeholder(dtype=tf.float64)

        self.tf_W = tf.Variable(dtype=tf.float64,
                                initial_value=tf.random.normal(dtype=tf.float64, shape=(self.M1, self.M2), mean=0,
                                                               stddev=1 / self.M1))
        self.tf_bh = tf.Variable(dtype=tf.float64, initial_value=tf.random.normal(dtype=tf.float64, shape=(self.M2,)))
        self.tf_bo = tf.Variable(dtype=tf.float64, initial_value=tf.random.normal(dtype=tf.float64, shape=(self.M1,)))

        tf_Xhat = self.forward(tf_X)
        tf_cost = tf.math.reduce_sum(tf.square(tf_X - tf_Xhat))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_cost)
        tf_Z = tf.nn.sigmoid(tf.math.add(tf.matmul(a=tf_X, b=self.tf_W), self.tf_bh))

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            iteration = 0
            iterations = []

            nBatches = np.int(np.round(N * 1.0 / batch_size - 0.5))

            for i in range(epoch):
                for j in range(nBatches + 1):
                    iterations.append(iteration)

                    # mini-batch gradient descent
                    if j == nBatches:
                        session.run(train_op, feed_dict={tf_X: X[j * nBatches:N]})
                    else:
                        session.run(train_op, feed_dict={tf_X: X[j * nBatches:(j + 1) * nBatches]})

                    if iteration % 50 == 0:
                        cost = session.run(tf_cost, feed_dict={tf_X: X})
                        print(
                            "|      Pretraining. Epoch " + str(i) + "/ Iteration " + str(
                                iteration) + "/ Training error = " + str(
                                cost / len(X)))

                    iteration += 1

            self.Z = session.run(tf_Z, feed_dict={tf_X: X, tf_Xhat: X})

    def forwardTraining(self, X):
        """
        Use in conjunction with other autoencoders
        :param X:
        :return:
        """
        return tf.nn.sigmoid(tf.math.add(tf.matmul(a=X, b=self.tf_W), self.tf_bh))

    def forward(self, X):
        """
        Forward to find optimal weights in this autoencoder
        :param X:
        :return:
        """
        Z = tf.nn.sigmoid(tf.math.add(tf.matmul(a=X, b=self.tf_W), self.tf_bh))
        X_hat = tf.nn.sigmoid(tf.math.add(tf.matmul(a=Z, b=tf.transpose(self.tf_W)), self.tf_bo))
        return X_hat


class ANN:
    """
    Represent a network: an input layer, hidden layers, and an output layer
    """

    def __init__(self, D, hiddenLayersSize, K):
        """
        :param D: number of features
        :param K: number of classes
        :param hiddenLayersSize: 1-D dimension, represents the number of units in each hidden layer
        """
        self.D = D
        self.K = K
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
        inputLayer = AutoEncoder(nFeatures, hiddenLayersSize[0])
        layers.append(inputLayer)

        for idx, numUnits in enumerate(hiddenLayersSize):

            if idx == len(hiddenLayersSize) - 1:
                hiddenLayer = OutputLayer(numUnits, nClasses)
                layers.append(hiddenLayer)
            else:
                hiddenLayer = AutoEncoder(numUnits, hiddenLayersSize[idx + 1])
                layers.append(hiddenLayer)
        return layers

    def fit(self, Xtrain, ytrain, learning_rate=0.001, epoch=20, batch_size=100):
        N = Xtrain.shape[0]
        self.layers = self.initializeLayers(self.D, self.K, self.hiddenLayersSize)

        # STEP 1: greedy layer-wise training of autoencoders
        input_autoencoder = Xtrain

        for layer in self.layers[:-1]:
            print('Pretraining layer = (' + str(layer.M1) + ', ' + str(layer.M2) + ')')
            layer.fit(input_autoencoder)
            input_autoencoder = layer.Z

        # STEP 2
        print('Fit model')
        self.tf_X = tf.placeholder(dtype=tf.float64)
        tf_Y = tf.placeholder(dtype=tf.float64)

        Ytrain = utils.convert2indicator(ytrain)

        self.tf_Yhat = self.forward(self.tf_X)
        tf_cost = tf.math.reduce_sum(- tf_Y * tf.math.log(self.tf_Yhat))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        iteration = 0
        iterations = []
        costs = []

        nBatches = np.int(np.round(N * 1.0 / batch_size - 0.5))

        for i in range(epoch):
            for j in range(nBatches + 1):
                iterations.append(iteration)

                # mini-batch gradient descent
                trainingCost = 0
                if j == nBatches:
                    _, trainingCost = self.session.run((train_op, tf_cost), feed_dict={
                        self.tf_X: Xtrain[j * nBatches:N],
                        tf_Y: Ytrain[j * nBatches:N]})
                else:
                    _, trainingCost = self.session.run((train_op, tf_cost), feed_dict={
                        self.tf_X: Xtrain[j * nBatches:(j + 1) * nBatches],
                        tf_Y: Ytrain[j * nBatches:(j + 1) * nBatches]})

                # just for testing
                costs.append(trainingCost)

                print("Training. Epoch " + str(i) + "/ Iteration " + str(iteration)
                      + "/ Training error = " + str(trainingCost / len(Xtrain)))

                iteration += 1

    def forward(self, X):
        for layer in self.layers:
            X = layer.forwardTraining(X)
        return X

    def predict(self, X):
        Yhat = self.session.run(self.tf_Yhat, feed_dict={self.tf_X: X})
        yhat = np.argmax(Yhat, axis=1)
        return yhat

    def score(self, y, yhat):
        return np.mean(y == yhat)


def main():
    # build model
    Xtrain, ytrain = utils.readTrainingDigitRecognizer('../data/digit-recognizer/train.csv')
    D = Xtrain[0].shape[0]
    ae = ANN(D=D, hiddenLayersSize=[64, 32, 16], K=10)
    ae.fit(Xtrain, ytrain)

    # Prediction
    Xtest = utils.readTestingDigitRecognizer('../data/digit-recognizer/test.csv')
    yhat = ae.predict(Xtest)
    print('Prediction: ' + str(yhat))

    # Export to csv
    import csv
    with open('../data/digit-recognizer/submission.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['ImageId', 'Label'])

        for idx, row in enumerate(yhat):
            writer.writerow([idx + 1, row])

    csvFile.close()


main()