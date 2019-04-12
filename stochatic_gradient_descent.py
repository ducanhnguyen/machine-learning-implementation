# Data: digit-recognizer (number of classes = 10)
# Objective: Compare learning rate strategy: constant, Adagrad, RMSProp, and step decay
# Method: Use deep learning, 1 hidden layer (M units), softmax activation

import matplotlib.pylab as plt
import numpy as np

import utils


class Neural_Network:
    def __init__(self, M):
        '''
        :param M: Number of units in the first layer
        '''
        self.M = M

    def initializeWeights(self, K, D, M):
        W1 = np.random.rand(D, M)
        b1 = np.random.rand(M)
        W2 = np.random.rand(M, K)
        b2 = np.random.rand(K)
        return W1, b1, W2, b2

    def updateW2andb2(self, W2, b2, Y, Y_hat, Z, N, K, M, startRange, endRange):
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

    def updateW1andb1(self, W1, W2, b1, X, D, Y, Y_hat, Z, N, K, M, startRange, endRange):
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

    def fit(self, Xtrain, ytrain, strategy):
        '''
        Build model
        :param Xtrain: a set of observations
        :param ytrain: labels
        :param epoch:
        :param learning_rate:
        :return:
        '''
        epoch = strategy['epoch']
        learning_rate = strategy['learning_rate']
        L2_regulation = strategy['L2_regulation']

        Y = utils.convert2indicator(ytrain)
        K = np.amax(ytrain) + 1  # get number of classes
        N, D = Xtrain.shape
        self.W1, self.b1, self.W2, self.b2 = self.initializeWeights(K, D, self.M)

        # for logging
        l_cost = list()
        l_iterations = list()
        l_score = list()

        iteration = -1

        self.cache_W2 = 1  # adagrad, rmsprop
        self.cache_b2 = 1  # adgrad, rmsprop
        self.cache_W1 = 1  # adgrad, rmsprop
        self.cache_b1 = 1  # adarad, rmsprop

        self.v_W1 = 0  # momentum
        self.v_b1 = 0  # momentum
        self.v_W2 = 0  # momentum
        self.v_b2 = 0  # momentum

        for i in range(0, epoch):

            # update weights
            for j in range(0, N):
                # for each epoch, run over all samples separately
                print('Epoch ' + str(i))
                iteration += 1
                print('Iteration ' + str(iteration))
                print('Learning rate: ' + str(learning_rate))

                # compute score
                Yhat, Z = self.predict(Xtrain)
                yhat = np.argmax(Yhat, axis=1)
                yindex = np.argmax(Y, axis=1)
                score = np.mean(yhat == yindex)
                l_score.append(score)
                print('Score: ' + str(score))

                cost = self.cost(Yhat, Y)
                l_cost.append(cost)
                l_iterations.append(iteration)
                print('Cost: ' + str(cost))

                # compute the gradient at a single observation
                startRange = j
                endRange = j + 1
                print('Choose observation ' + str(startRange))
                gradient_W2, gradient_b2 = self.updateW2andb2(self.W2, self.b2, Y, Yhat, Z, N, K, self.M,
                                                              startRange,
                                                              endRange)
                gradient_W1, gradient_b1 = self.updateW1andb1(self.W1, self.W2, self.b1, Xtrain, D, Y, Yhat, Z, N, K,
                                                              self.M,
                                                              startRange,
                                                              endRange)
                print('Update weights')

                # update learning rate
                if strategy['name'] == 'STEP_DECAY':
                    if iteration >= 1 and iteration % strategy['step'] == 0:
                        learning_rate = learning_rate / strategy['factor']

                    self.W1 += learning_rate * (gradient_W1 + L2_regulation * self.W1)
                    self.b1 += learning_rate * (gradient_b1 + L2_regulation * self.b1)
                    self.W2 += learning_rate * (gradient_W2 + L2_regulation * self.W2)
                    self.b2 += learning_rate * (gradient_b2 + L2_regulation * self.b2)

                elif strategy['name'] == 'ADAGRAD':
                    self.cache_b1 += gradient_b1 * gradient_b1
                    self.b1 += learning_rate * (gradient_b1 + L2_regulation * self.b1) / (np.sqrt(
                        self.cache_b1) + strategy['epsilon'])

                    self.cache_W1 += gradient_W1 * gradient_W1
                    self.W1 += learning_rate * (gradient_W1 + L2_regulation * self.W1) / (np.sqrt(
                        self.cache_W1) + strategy['epsilon'])

                    self.cache_b2 += gradient_b2 * gradient_b2
                    self.b2 += learning_rate * (gradient_b2 + L2_regulation * self.b2) / (np.sqrt(
                        self.cache_b2) + strategy['epsilon'])

                    self.cache_W2 += gradient_W2 * gradient_W2
                    self.W2 += learning_rate * (gradient_W2 + L2_regulation * self.W2) / (np.sqrt(
                        self.cache_W2) + strategy['epsilon'])

                elif strategy['name'] == 'CONSTANT':
                    self.W1 += learning_rate * (gradient_W1 + L2_regulation * self.W1)
                    self.b1 += learning_rate * (gradient_b1 + L2_regulation * self.b1)
                    self.W2 += learning_rate * (gradient_W2 + L2_regulation * self.W2)
                    self.b2 += learning_rate * (gradient_b2 + L2_regulation * self.b2)

                elif strategy['name'] == 'RMSPROP':
                    self.cache_b1 = strategy['decay_rate'] * self.cache_b1 + (
                            1 - strategy['decay_rate']) * gradient_b1 * gradient_b1
                    self.b1 += learning_rate * (gradient_b1 + L2_regulation * self.b1) / (np.sqrt(
                        self.cache_b1) + strategy['epsilon'])

                    self.cache_W1 = strategy['decay_rate'] * self.cache_W1 + (
                            1 - strategy['decay_rate']) * gradient_W1 * gradient_W1
                    self.W1 += learning_rate * (gradient_W1 + L2_regulation * self.W1) / (np.sqrt(
                        self.cache_W1) + strategy['epsilon'])

                    self.cache_b2 = strategy['decay_rate'] * self.cache_b2 + (
                            1 - strategy['decay_rate']) * gradient_b2 * gradient_b2
                    self.b2 += learning_rate * (gradient_b2 + L2_regulation * self.b2) / (np.sqrt(
                        self.cache_b2) + strategy['epsilon'])

                    self.cache_W2 = strategy['decay_rate'] * self.cache_W2 + (
                            1 - strategy['decay_rate']) * gradient_W2 * gradient_W2
                    self.W2 += learning_rate * (gradient_W2 + L2_regulation * self.W2) / (np.sqrt(
                        self.cache_W2) + strategy['epsilon'])

                elif strategy['name'] == 'MOMENTUM':
                    self.v_W1 = strategy['mu'] * self.v_W1 + learning_rate * (gradient_W1 + L2_regulation * self.W1)
                    self.W1 += self.v_W1

                    self.v_b1 = strategy['mu'] * self.v_b1 + learning_rate * (gradient_b1 + L2_regulation * self.b1)
                    self.b1 += self.v_b1

                    self.v_W2 = strategy['mu'] * self.v_W2 + learning_rate * (gradient_W2 + L2_regulation * self.W2)
                    self.W2 += self.v_W2

                    self.v_b2 = strategy['mu'] * self.v_b2 + learning_rate * (gradient_b2 + L2_regulation * self.b2)
                    self.b2 += self.v_b2

                print('\n')

        return l_cost, l_iterations, l_score

    def predict(self, X):
        Z = utils.softmax(X.dot(self.W1) + self.b1)  # Z:(N, M)
        Yhat = utils.softmax(Z.dot(self.W2) + self.b2)
        return Yhat, Z

    def cost(self, Yhat, Y):
        '''
        Cross entropy cost
        :param Yhat: the prediction over classes
        :param Y: the true distribution
        :return:
        '''
        cost = 0
        for idx, yhat in enumerate(Yhat):
            for class_index, predicted_class_probability in enumerate(yhat):
                cost += Y[idx][class_index] * np.log(predicted_class_probability)

        cost = -1 * cost
        return cost


def main():
    ANALYZED_OBSERVATIONS = 1000
    Xtrain, ytrain = utils.readTrainingDigitRecognizer('./data/digit-recognizer/train.csv', limit=ANALYZED_OBSERVATIONS)
    s = Neural_Network(M=7)

    # global configuration
    LEARNING_RATE = 0.01
    EPOCH = 2

    strategies = {
        'strategy_0': {'name': 'STEP_DECAY', 'factor': 2, 'step': 1000, 'learning_rate': LEARNING_RATE, 'epoch': EPOCH,
                       'L2_regulation': 0},
        'strategy_1': {'name': 'ADAGRAD', 'epsilon': 10e-8, 'learning_rate': LEARNING_RATE, 'epoch': EPOCH,
                       'L2_regulation': 0},
        'strategy_2': {'name': 'CONSTANT', 'learning_rate': LEARNING_RATE, 'epoch': EPOCH, 'L2_regulation': 0},
        'strategy_3': {'name': 'RMSPROP', 'epsilon': 10e-8, 'decay_rate': 0.99, 'learning_rate': LEARNING_RATE,
                       'epoch': EPOCH, 'L2_regulation': 0},
        'strategy_4': {'name': 'MOMENTUM', 'mu': 0.9, 'learning_rate': LEARNING_RATE, 'epoch': EPOCH,
                       'L2_regulation': 0}
    }

    for _, strategy in strategies.items():
        l_cost, l_iterations, l_score = s.fit(Xtrain, ytrain, strategy=strategy)
        plt.plot(l_iterations, l_score, label=str(strategy))

    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(
        'Gradient descent variants comparison \n(data = digit-recognizer, ' +
        'observations = ' + str(ANALYZED_OBSERVATIONS) + ' )')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
