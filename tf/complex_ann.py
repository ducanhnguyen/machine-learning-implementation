"""
# Data: digit-recognizer (number of classes = 10)
# Method: deep learning, softmax activation on all layers
# Use tensorflow
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

import utils


class Layer:
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        self.tf_W = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M1, M2), mean=0, stddev=1))
        self.tf_b = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal(shape=(M2,)))
        self.weights = [self.tf_W, self.tf_b]


class NeuralNetwork:
    def __init__(self, hiddenLayersSize):
        self.hiddenLayersSize = hiddenLayersSize

    def initializeLayers(self, nFeatures, nClasses, hiddenLayersSize):
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

    def fit(self, X, y, epoch=1000):
        K = np.amax(y) + 1  # get the number of classes
        Y = utils.convert_to_indicator(y)
        N, D = X.shape

        layers = self.initializeLayers(nFeatures=D, nClasses=K, hiddenLayersSize=self.hiddenLayersSize)

        # initialize placeholders
        tf_X = tf.placeholder(dtype=tf.float32, name='X', shape=(N, D))
        tf_Y = tf.placeholder(dtype=tf.float32, name='Y', shape=(N, K))

        # define symbolic formula
        tf_Yhat = self.forward(tf_X, layers)  # backpropogation
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

                yhat = session.run(tf_yhat, feed_dict={tf_X: X, tf_Y: Y})
                score = np.mean(yhat == y)
                print('score: ' + str(score))

                iterations.append(i)
                scores.append(score)

                print()

        self.plot(scores, iterations)

    def forward(self, tf_X, layers):
        tf_input = tf_X
        for layer in layers:
            tf_input = tf.nn.softmax(tf.matmul(tf_input, layer.tf_W) + layer.tf_b)

        return tf_input

    def plot(self, scores, iterations):
        plt.plot(iterations, scores)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Learning rate comparison (data = digit-recognizer)')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    X_train, y_train = utils.read_csv('../data/digit-recognizer/train.csv', limit=1000)
    s = NeuralNetwork([10, 5])
    s.fit(X_train, y_train)


if __name__ == "__main__":
    main()
