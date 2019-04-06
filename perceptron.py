# data: https://www.kaggle.com/c/digit-recognizer
# Method: Perceptron
# For simplicity, we take binary classification with class 0 and class 1
import pandas as pd
import numpy as np
import utils


def filter(X, Y):
    '''
    only keep the observations belonging to class 0 or class 1
    :param X:
    :param Y:
    :return:
    '''
    X2 = list()
    Y2 = list()
    for idx, value in enumerate(Y):
        if value == 0 or value == 1:
            X2.append(X[idx])

            if value == 0:
                Y2.append(-1)
            else:
                Y2.append(1)
    return X2, Y2


class Perceptron:

    def fit(self, X, y, epoch=100, learning_rate=0.01):
        X = np.array(X)
        y = np.array(y)

        # Initialize w
        row = X.shape[1]
        self.w = np.zeros(row)
        self.b = 0

        # build model here
        for i in range(epoch):
            y_hat = self.predict(X)

            misclassified = list()
            for idx in range(len(y_hat)):
                if y[idx] != y_hat[idx]:
                    misclassified.append(idx)

            print("Epoch = " + str(i) + ": num of misclassified points = " + str(len(misclassified)))

            if len(misclassified) == 0:
                break
            else:
                # choose a randomly misclassified point
                idx_random = np.random.choice(misclassified)
                self.w += learning_rate * X[idx_random] * y[idx_random]
                self.b += learning_rate * y[idx_random]
        return self.w, self.b

    def predict(self, X):
        X = np.array(X)
        return np.sign(X.dot(self.w) + self.b)

    def score(selfs, y_hat, y):
        return np.mean(y_hat == y)


def main():
    X, Y = utils.read_csv('./data/digit-recognizer/train.csv', limit=2000)
    X, Y = filter(X, Y)
    N = int(len(X) / 2)
    X_train = X[:N]
    y_train = Y[:N]
    X_test = X[N:]
    y_test = Y[N:]

    p = Perceptron()
    p.fit(X_train, y_train, epoch=5000)
    print("Training done.")

    y_hat = p.predict(X_test)
    print("Use test data. Prediction done!")
    score = p.score(y_hat, y_test)

    print("Accuracy = " + str(score))


if __name__ == "__main__":
    main()
