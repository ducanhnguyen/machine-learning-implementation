# data: https://www.kaggle.com/c/digit-recognizer
# Method: K-nearest neighbour

import numpy as np
import matplotlib.pylab as plt
import utils

class KNN:
    def __init__(self, k):
        '''
        Initialize KNN
        :param k: the number of closest neighours
        '''
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        y_hat = np.zeros(len(X_test))

        for idx, x in enumerate(X_test):
            # find k-nearest points
            l_nearest_points = list()  # (distance, class)*

            for idx_2, x_2 in enumerate(self.X):
                diff = x - x_2
                distance = diff.dot(diff)

                if distance > 0:
                    if (len(l_nearest_points) < self.k):
                        l_nearest_points.append((distance, self.y[idx_2]))
                    else:
                        # remove the point having the highest distance
                        largest = 0.0
                        idx_largest = 0
                        for idx_tmp, nearest_point in enumerate(l_nearest_points):
                            if nearest_point[0] > largest:
                                largest = nearest_point[0]
                                idx_largest = idx_tmp

                        if distance < l_nearest_points[idx_largest][0]:
                            l_nearest_points[idx_largest] = (distance, self.y[idx_2])

            # voting
            c = dict()  # (class, num of occurrence)*
            for _, point in l_nearest_points:
                if point in c:
                    c[point] = c[point] + 1
                else:
                    c[point] = 1

            import operator
            sorted_c = sorted(c.items(), key=operator.itemgetter(1))
            y_hat[idx] = sorted_c[len(sorted_c) - 1][0]

        return y_hat

    def score(self, y_hat, y):
        return np.mean(y == y_hat)


def main():
    X, y = utils.read_csv('./data/digit-recognizer/train.csv', limit=2000)
    TRAIN = 1000
    X_train = X[:TRAIN]
    y_train = y[:TRAIN]
    X_test = X[TRAIN:]
    y_test = y[TRAIN:]

    dict_accuracies = dict()
    K = 10  # num of nearest neighbours
    for k in range(1, K):
        knn = KNN(k)
        knn.fit(X=X_train, y=y_train)
        y_hat = knn.predict(X_test=X_test)
        accuracy = knn.score(y_hat=y_hat, y=y_test)
        print("k = " + str(k) + ": accuracy = " + str(accuracy))
        dict_accuracies[k] = accuracy

    # Plot
    lists = sorted(dict_accuracies.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()