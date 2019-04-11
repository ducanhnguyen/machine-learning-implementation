"""
Implementation of AutoEncoder with 1 hidden layer. I try to encode images, and then decode these images to get the original ones.
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

import utils


class AutoEncoder:
    def __init__(self, M):
        """

        :param M: Number of units in the hidden layer
        """
        self.M = M

    def fit(self, X, learning_rate=0.01, epoch=100, batch_size=50):
        N, D = X.shape
        tf_X = tf.placeholder(dtype=tf.float32)
        self.tf_W1 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(D, self.M), mean=0, stddev=1))
        self.tf_b1 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(self.M,)))

        self.tf_W2 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(self.M, D), mean=0, stddev=1))
        self.tf_b2 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(D,)))

        tf_Xhat = self.forward(tf_X)
        tf_cost = tf.math.reduce_sum(tf.square(tf_X - tf_Xhat))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_cost)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            iteration = 0
            iterations = []
            costs = []

            nBatches = np.int(np.round(N * 1.0 / batch_size - 0.5))

            for i in range(epoch):
                for j in range(nBatches + 1):
                    iterations.append(iteration)

                    # mini-batch gradient descent
                    if j == nBatches:
                        session.run(train_op, feed_dict={tf_X: X[j * nBatches:N]})
                    else:
                        session.run(train_op, feed_dict={tf_X: X[j * nBatches:(j + 1) * nBatches]})

                    cost = session.run(tf_cost, feed_dict={tf_X: X})
                    print("Epoch " + str(i) + "/ Iteration " + str(iteration) + "/ Cost = " + str(cost))
                    costs.append(cost)

                    iteration += 1

            self.plotCost(iterations, costs)

            # show the first image
            Xhat = session.run(tf_Xhat, feed_dict={tf_X: X})
            self.plotComparison(X[0], Xhat[0])

    def forward(self, X):
        Z = tf.nn.sigmoid(tf.math.add(tf.matmul(a=X, b=self.tf_W1), self.tf_b1))
        X_hat = tf.nn.sigmoid(tf.math.add(tf.matmul(a=Z, b=self.tf_W2), self.tf_b2))
        return X_hat

    def plotComparison(self, x, xhat):
        # original
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')

        # reconstruction

        plt.subplot(1, 2, 2)
        plt.imshow(xhat.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.show()

    def plotCost(self, iterations, costs):
        """
        Visualization
        :param scores: 1-D dimension of float numbers
        :param iterations:  1-D dimension of integer numbers
        :return:
        """
        plt.plot(iterations, costs, label="cost over iteration")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Autoencoder (data = digit-recognizer)')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    X, Y = utils.readCsv('../data/digit-recognizer/train.csv', limit=10)
    ae = AutoEncoder(M=16)
    ae.fit(X, learning_rate=0.01, epoch=10000, batch_size=50)


main()
