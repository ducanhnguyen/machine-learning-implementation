# Data: digit-recognizer (number of classes = 10)
# Objective: Play with different learning rate
# Method: deep learning, 1 hidden layer (M units), softmax activation, full gradient ascent
# Note: Because we need to maximize the objective function, therefore, we must use gradient ascent to find local optimal solution
# Result:
# - If we use too large learning rate (e.g., 0.1), the problem of NaN happens.
# - With three learning rate (i.e., 0.0001, 0.00001, 0.000001), the first learning rate converges faster. The last learing rate converges slowest.

import numpy as np
import matplotlib.pylab as plt
import utils

class Neural_Network:
    def __init__(self, M):
        '''
        :param M: Number of units in the first layer
        '''
        self.M = M

    def initialize_weight(self, K, D, M):
        W1 = np.random.rand(D, M)
        b1 = np.random.rand(M)
        W2 = np.random.rand(M, K)
        b2 = np.random.rand(K)
        return W1, b1, W2, b2

    def update_W2_and_b2(self, W2, b2, Y, Y_hat, Z, N, K, M, startRange, endRange):
        # very slow
        '''
        gradient_w2 = np.zeros((W2.shape[0], W2.shape[1]))
        gradient_b2 = np.zeros((b2.shape[0]))
        for n in range(startRange, endRange):
            for k in range(0, K):
                gradient_b2[k] += Y[n][k] - Y_hat[n][k]

                for m in range(0, M):
                    gradient_w2[m][k] +=(Y[n][k] - Y_hat[n][k]) * Z[n][m]
        '''
        # faster
        gradient_w2 = Z[startRange:endRange].T.dot(Y[startRange:endRange] - Y_hat[startRange:endRange])
        gradient_b2 = np.sum(Y[startRange:endRange] - Y_hat[startRange:endRange], axis=0)
        return gradient_w2, gradient_b2

    def update_W1_and_b1(self, W1, W2, b1, X, D, Y, Y_hat, Z, N, K, M, startRange, endRange):
        # very slow
        '''
        gradient_w1 = np.zeros((W1.shape[0], W1.shape[1]))
        gradient_b1 = np.zeros((b1.shape[0]))

        for n in range(startRange, endRange):

            for k in range(0, K):

                for m in range(0, M):
                    gradient_b1[m] += (Y[n][k] - Y_hat[n][k]) * W2[m][k] * Z[n][m] * (1 - Z[n][m])

                    for d in range(0, D):
                        gradient_w1[d][m] += gradient_b1[m] * X[n][d]
        '''
        # faster
        dZ = (Y[startRange:endRange] - Y_hat[startRange:endRange]).dot(W2.T) * Z[startRange:endRange] * (
                1 - Z[startRange:endRange])
        gradient_w1 = X[startRange:endRange].T.dot(dZ)

        gradient_b1 = np.sum(
            (Y[startRange:endRange] - Y_hat[startRange:endRange]).dot(W2.T) * Z[startRange:endRange] * (1 - Z)[
                                                                                                       startRange:endRange],
            axis=0)
        return gradient_w1, gradient_b1

    def fit(self, X, y, epoch=1000, learning_rate=0.01, L2_regulation=0.0):
        '''
        Build model
        :param X: a set of observations
        :param y: labels
        :param epoch:
        :param learning_rate:
        :return:
        '''
        K = np.amax(y) + 1
        Y = utils.convert_to_indicator(y)
        N, D = X.shape
        self.W1, self.b1, self.W2, self.b2 = self.initialize_weight(K, D, self.M)

        l_cost = list()
        l_iterations = list()
        l_score = list()

        for i in range(0, epoch):
            # update weights
            print('Epoch ' + str(i))

            # compute score
            Y_hat, Z = self.predict(X)
            y_hat = np.argmax(Y_hat, axis=1)
            y_index = np.argmax(Y, axis=1)
            score = np.mean(y_hat == y_index)
            l_score.append(score)
            print('Score: ' + str(score))

            cost = self.cost(Y_hat, Y)
            l_cost.append(cost)
            l_iterations.append(i)
            print('Cost: ' + str(cost))

            # full gradient descent
            # compute the gradient over the whole data
            startRange = 0
            endRange = N

            gradient_W2, gradient_b2 = self.update_W2_and_b2(self.W2, self.b2, Y, Y_hat, Z, N, K, self.M,
                                                             startRange,
                                                             endRange)
            gradient_W1, gradient_b1 = self.update_W1_and_b1(self.W1, self.W2, self.b1, X, D, Y, Y_hat, Z, N, K,
                                                             self.M,
                                                             startRange,
                                                             endRange)

            self.W1 += learning_rate * (gradient_W1 + L2_regulation * self.W1)
            self.b1 += learning_rate * (gradient_b1 + L2_regulation * self.b1)
            self.W2 += learning_rate * (gradient_W2 + L2_regulation * self.W2)
            self.b2 += learning_rate * (gradient_b2 + L2_regulation * self.b2)
            print('\n')

        return l_cost, l_iterations, l_score

    def predict(self, X):
        # print('Computing Z')
        Z = utils.softmax(X.dot(self.W1) + self.b1)  # Z:(N, M)

        # print('Computing Y_hat')
        Y_hat = utils.softmax(Z.dot(self.W2) + self.b2)
        return Y_hat, Z

    def cost(self, Y_hat, Y):
        '''
        Cross entropy cost
        :param Y_hat: the prediction over classes
        :param Y: the true distribution
        :return:
        '''
        cost = 0
        for idx, y_hat in enumerate(Y_hat):
            for class_index, predicted_class_probability in enumerate(Y_hat[idx]):
                cost += Y[idx][class_index] * np.log(predicted_class_probability)

        cost = -1 * cost
        return cost

def main():
    X_train, y_train = utils.read_csv('./data/digit-recognizer/train.csv', limit=10000)
    s = Neural_Network(M=7)

    # plot score with different learning rates
    l_cost_0, l_iterations_0, l_score_0 = s.fit(X_train, y_train, epoch=10, learning_rate=0.0001, L2_regulation=0)
    plt.plot(l_iterations_0, l_score_0, label='learning_rate=0.0001')

    l_cost_1, l_iterations_1, l_score_1 = s.fit(X_train, y_train, epoch=10, learning_rate=0.00001, L2_regulation=0)
    plt.plot(l_iterations_1, l_score_1, label='learning_rate=0.00001')

    l_cost_2, l_iterations_2, l_score_2 = s.fit(X_train, y_train, epoch=10, learning_rate=0.000001, L2_regulation=0)
    plt.plot(l_iterations_2, l_score_2, label='learning_rate=0.000001')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Learning rate comparison (data = digit-recognizer, size = ' + str(len(X_train)) + ")")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
