"""
# Data: digit-recognizer (number of classes = 10)
# Method: deep learning, 1 hidden layer (M units), softmax activation, full gradient ascent
# Use tensorflow
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

import utils


class NeuralNetwork:
    def __init__(self, M):
        '''
        :param M: Number of units in the first layer
        '''
        self.M = M

    def initializeWeights(self, D, M, K):
        tf_W1 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(D, M), mean=0, stddev=1), name='W1')
        tf_b1 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M,)), name='b1')

        tf_W2 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M, K), mean=0, stddev=1), name='W2')
        tf_b2 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(K,)), name='b2')

        return tf_W1, tf_b1, tf_W2, tf_b2

    def initializePlaceholder(self):
        tf_X = tf.placeholder(dtype=tf.float32, name='X')
        tf_Y = tf.placeholder(dtype=tf.float32, name='Y')
        return tf_X, tf_Y

    def fit(self, X, y, epoch=1000):
        K = np.amax(y) + 1
        Y = utils.convert2indicator(y)
        N, D = X.shape

        tf_X, tf_Y = self.initializePlaceholder()
        tf_W1, tf_b1, tf_W2, tf_b2 = self.initializeWeights(D, self.M, K)

        # define symbolic operations
        tf_Yhat = self.forward(tf_X, tf_W1, tf_b1, tf_W2, tf_b2)
        tf_cost = tf.math.reduce_sum(-1 * tf.multiply(tf_Y, tf.log(tf_Yhat)))  # cross-entropy
        tf_train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(tf_cost)
        tf_yhat = tf.math.argmax(tf_Yhat, axis=1)

        scores = []
        iterations = []

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for i in range(epoch):
                print('iteration ' + str(i))
                session.run(tf_train, feed_dict={tf_X: X, tf_Y: Y})

                cost = session.run(tf_cost, feed_dict={tf_X: X, tf_Y: Y})
                print("Cost = " + str(cost))

                yhat = session.run(tf_yhat, feed_dict={tf_X: X, tf_Y: Y})
                score = np.mean(yhat == y)
                print('Score = ' + str(score))

                iterations.append(i)
                scores.append(score)

                print()

        self.plot(scores, iterations)

    def plot(self, scores, iterations):
        plt.plot(iterations, scores)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Error (data = digit-recognizer)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def forward(self, X, tf_W1, tf_b1, tf_W2, tf_b2):
        tf_Z = tf.nn.softmax(tf.matmul(X, tf_W1) + tf_b1)
        tf_Yhat = tf.nn.softmax(tf.matmul(tf_Z, tf_W2) + tf_b2)
        return tf_Yhat


def main():
    X_train, y_train = utils.readCsv('../data/digit-recognizer/train.csv', limit=1000)
    s = NeuralNetwork(M=7)
    s.fit(X_train, y_train)


if __name__ == "__main__":
    main()
