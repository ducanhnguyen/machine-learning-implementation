# data: https://www.kaggle.com/c/digit-recognizer
# Method: K-nearest neighbour

import matplotlib.pylab as plt
import numpy as np

import utils


class KNN:
    def __init__(self, k):
        '''
        Initialize KNN
        :param k: the number of closest neighours
        '''
        self.k = k

    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain

    def predict(self, Xtest):
        yhat = np.zeros(len(Xtest))

        for idx, x in enumerate(Xtest):
            # find k-nearest points
            l_nearestPoints = list()  # (distance, class)*

            for idx_2, x_2 in enumerate(self.Xtrain):
                diff = x - x_2
                distance = diff.dot(diff)

                if distance > 0:
                    if (len(l_nearestPoints) < self.k):
                        l_nearestPoints.append((distance, self.ytrain[idx_2]))
                    else:
                        # remove the point having the highest distance
                        largest = 0.0
                        idx_largest = 0
                        for idx_tmp, nearest_point in enumerate(l_nearestPoints):
                            if nearest_point[0] > largest:
                                largest = nearest_point[0]
                                idx_largest = idx_tmp

                        if distance < l_nearestPoints[idx_largest][0]:
                            l_nearestPoints[idx_largest] = (distance, self.ytrain[idx_2])

            # voting
            c = dict()  # (class, num of occurrence)*
            for _, point in l_nearestPoints:
                if point in c:
                    c[point] = c[point] + 1
                else:
                    c[point] = 1

            import operator
            sorted_c = sorted(c.items(), key=operator.itemgetter(1))
            yhat[idx] = sorted_c[len(sorted_c) - 1][0]

        return yhat

    def score(self, yhat, y):
        return np.mean(y == yhat)


def main():
    X, y = utils.readTrainingDigitRecognizer('./data/digit-recognizer/train.csv', limit=2000)
    TRAIN = 1500
    Xtrain = X[:TRAIN]
    ytrain = y[:TRAIN]
    Xval = X[TRAIN:]
    yval = y[TRAIN:]

    accuracies = dict()
    K = 10  # num of nearest neighbours
    for k in range(1, K):
        knn = KNN(k)
        knn.fit(Xtrain=Xtrain, ytrain=ytrain)
        yhat = knn.predict(Xtest=Xval)
        accuracy = knn.score(yhat=yhat, y=yval)
        print("k = " + str(k) + ": accuracy = " + str(accuracy))
        accuracies[k] = accuracy

    # Plot
    lists = sorted(accuracies.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, label = 'K-nearest neighbour')
    plt.xlabel('number of nearest points')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == "__main__":
    main()
