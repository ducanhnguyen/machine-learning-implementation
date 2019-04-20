"""
Implementation of RBM.

Unlike autoencoder, which may have more than 1 hidden layer, RBM has only one hidden layer.
I try to encode images, and then decode these images to get the original ones.
"""

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

import utils


class RBM:
    def __init__(self, M):
        """
        :param M: Number of units in the hidden layer
        """
        self.M = M

    def fit(self, X, learning_rate=0.001, epoch=10, batch_size=100):
        N, D = X.shape
        tf_V = tf.placeholder(dtype=tf.float32, shape=(None, D))

        self.tf_W = tf.Variable(dtype=tf.float32,
                                initial_value=tf.random.normal(shape=(D, self.M), mean=0,
                                                               stddev=1 / D))  # xavier initialization
        self.tf_c = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(self.M, 1)))
        self.tf_b = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(D, 1)))

        tf_Vhat = self.forward(tf_V)
        tf_Vsample = self.Gibbs_sample(tf_V)

        tf_cost = tf.reduce_mean(self.F(tf_V, self.M) - self.F(tf_Vsample, self.M))

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

            self.plotCost(iterations, costs)

            # show images
            Xhat = session.run(tf_Vhat, feed_dict={tf_V: X})
            self.plotComparison(X[0], Xhat[0])
            self.plotComparison(X[1], Xhat[1])
            self.plotComparison(X[2], Xhat[2])
            self.plotComparison(X[3], Xhat[3])

    def Gibbs_sample(self, V):
        p_h_given_v = tf.nn.sigmoid(tf.math.add(tf.matmul(V, self.tf_W), tf.transpose(self.tf_c)))
        r = tf.random_uniform(shape=tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)

        p_v_given_h = tf.nn.sigmoid(tf.math.add(tf.matmul(H, tf.transpose(self.tf_W)), tf.transpose(self.tf_b)))
        r = tf.random_uniform(shape=tf.shape(p_v_given_h))
        Vsample = tf.to_float(r < p_v_given_h)

        return Vsample

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
    X, y = utils.readTrainingDigitRecognizer('../data/digit-recognizer/train.csv')
    ae = RBM(M=300)
    ae.fit(X)


main()
