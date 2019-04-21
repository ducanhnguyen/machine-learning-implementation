"""
Implementation of AutoEncoder with more than hidden layer.

Kaggle: ~94.62%
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

        self.tf_W = tf.Variable(dtype=tf.float32,
                                initial_value=tf.random.normal(dtype=tf.float32, shape=(self.M, self.K), mean=0,
                                                               stddev=1 / self.M))
        self.tf_b = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(dtype=tf.float32, shape=(self.K,)))

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

    def forwardTraining(self, X):
        return tf.nn.softmax(tf.math.add(tf.matmul(a=X, b=self.tf_W), self.tf_b))


class RBM:
    def __init__(self, M1, M2):
        """
        :param M2: Number of units in the hidden layer
        """
        self.M1 = M1
        self.M2 = M2

    def fit(self, X, learning_rate=0.001, epoch=10, batch_size=100):
        N, D = X.shape
        tf_V = tf.placeholder(dtype=tf.float32, shape=(None, self.M1))

        self.tf_W = tf.Variable(dtype=tf.float32,
                                initial_value=tf.random.normal(dtype=tf.float32, shape=(self.M1, self.M2), mean=0,
                                                               stddev=1 / self.M1))  # xavier initialization
        self.tf_c = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(dtype=tf.float32, shape=(self.M2, 1)))
        self.tf_b = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(dtype=tf.float32, shape=(self.M1, 1)))

        tf_Vhat = self.forward(tf_V)
        tf_Vsample = self.Gibbs_sampler(tf_V)

        tf_cost = tf.reduce_mean(self.F(tf_V, self.M2) - self.F(tf_Vsample, self.M2))
        tf_Z = tf.nn.sigmoid(tf.math.add(tf.matmul(tf_V, self.tf_W), tf.transpose(self.tf_c)))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_cost)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            iteration = 0
            iterations = []
            costs = []

            nBatches = np.int(np.round(N * 1.0 / batch_size - 0.5))

            for i in range(epoch):
                for j in range(nBatches + 1):

                    # mini-batch gradient descent
                    cost = 0
                    if j == nBatches:
                        _ = session.run(train_op, feed_dict={tf_V: X[j * nBatches:N]})
                    else:
                        _ = session.run(train_op, feed_dict={tf_V: X[j * nBatches:(j + 1) * nBatches]})

                    if (iteration % 100 == 0):
                        iterations.append(iteration)

                        cost = session.run(tf_cost, feed_dict={tf_V: X})
                        print("Epoch " + str(i) + "/ Iteration " + str(iteration) + "/ Cost = " + str(cost))
                        costs.append(cost)

                    iteration += 1

            self.Z = session.run(tf_Z, feed_dict={tf_V: X})

    def Gibbs_sampler(self, V):
        p_h_given_v = tf.nn.sigmoid(tf.math.add(tf.matmul(V, self.tf_W), tf.transpose(self.tf_c)))
        Hsamples = self.Bernoulli_sampler(p_h_given_v)

        p_v_given_h = tf.nn.sigmoid(tf.math.add(tf.matmul(Hsamples, tf.transpose(self.tf_W)), tf.transpose(self.tf_b)))
        Vsamples = self.Bernoulli_sampler(p_v_given_h)

        return Vsamples

    def Bernoulli_sampler(self, p_success):
        r = tf.random_uniform(dtype=tf.float32, shape=tf.shape(p_success))
        samplers = tf.to_float(r < p_success)
        return samplers

    def F(self, tf_V, M):
        F = -tf.matmul(tf_V, self.tf_b)

        F -= tf.math.log(1 + tf.math.exp(
            tf.matmul(tf_V, self.tf_W)
            + tf.transpose(self.tf_c)))
        return F

    def forward(self, V):
        tf_H = tf.nn.sigmoid(tf.math.add(tf.matmul(V, self.tf_W), tf.transpose(self.tf_c)))
        tf_Vhat = tf.nn.sigmoid(tf.math.add(tf.matmul(tf_H, tf.transpose(self.tf_W)), tf.transpose(self.tf_b)))
        return tf_Vhat

    def forwardTraining(self, V):
        tf_H = tf.nn.sigmoid(tf.math.add(tf.matmul(a=V, b=self.tf_W), tf.transpose(self.tf_c)))
        return tf_H


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
        inputLayer = RBM(nFeatures, hiddenLayersSize[0])
        layers.append(inputLayer)

        for idx, numUnits in enumerate(hiddenLayersSize):

            if idx == len(hiddenLayersSize) - 1:
                hiddenLayer = OutputLayer(numUnits, nClasses)
                layers.append(hiddenLayer)
            else:
                hiddenLayer = RBM(numUnits, hiddenLayersSize[idx + 1])
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
        self.tf_X = tf.placeholder(dtype=tf.float32)
        tf_Y = tf.placeholder(dtype=tf.float32)

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
                # mini-batch gradient descent
                if j == nBatches:
                    self.session.run(train_op, feed_dict={
                        self.tf_X: Xtrain[j * nBatches:N],
                        tf_Y: Ytrain[j * nBatches:N]})
                else:
                    self.session.run(train_op, feed_dict={
                        self.tf_X: Xtrain[j * nBatches:(j + 1) * nBatches],
                        tf_Y: Ytrain[j * nBatches:(j + 1) * nBatches]})

                # just for testing
                if (iteration % 100 == 0):
                    iterations.append(iteration)

                    trainingCost = self.session.run(tf_cost, feed_dict={self.tf_X: Xtrain, tf_Y: Ytrain})
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
