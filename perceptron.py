# data: https://www.kaggle.com/c/digit-recognizer
# Method: Perceptron
# For simplicity, we take binary classification with class 0 and class 1
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

    def fit(self, Xtrain, ytrain, epoch=100, learning_rate=0.01):
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        # Initialize w
        row = Xtrain.shape[1]
        self.w = np.zeros(row)
        self.b = 0

        # build model here
        for i in range(epoch):
            yhat = self.predict(Xtrain)

            misclassified = list()
            for idx in range(len(yhat)):
                if ytrain[idx] != yhat[idx]:
                    misclassified.append(idx)

            print("Epoch = " + str(i) + ": num of misclassified points = " + str(len(misclassified)))

            if len(misclassified) == 0:
                break
            else:
                # choose a randomly misclassified point
                idx_random = np.random.choice(misclassified)
                self.w += learning_rate * Xtrain[idx_random] * ytrain[idx_random]
                self.b += learning_rate * ytrain[idx_random]
        return self.w, self.b

    def predict(self, X):
        X = np.array(X)
        return np.sign(X.dot(self.w) + self.b)

    def score(selfs, yhat, y):
        return np.mean(yhat == y)


def main():
    X, Y = utils.readCsv('./data/digit-recognizer/train.csv', limit=2000)
    X, Y = filter(X, Y)
    TRAIN = int(len(X) / 2)
    Xtrain = X[:TRAIN]
    ytrain = Y[:TRAIN]
    Xtest = X[TRAIN:]
    ytest = Y[TRAIN:]

    p = Perceptron()
    p.fit(Xtrain, ytrain, epoch=5000)
    print("Training done.")

    yhat = p.predict(Xtest)
    print("Use test data. Prediction done!")
    score = p.score(yhat, ytest)

    print("Accuracy = " + str(score))


if __name__ == "__main__":
    main()
