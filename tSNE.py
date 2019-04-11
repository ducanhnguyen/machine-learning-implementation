"""
tSNE (2008) is non-linear transformation. Unlike PCS, which is a linear transformation, tSANE has no model
transformation. Therefore, tSNE can not be used for classification and clustering. tSNE is only used for
visualization in 2-D or 3-D dimension from high dimensional data point.

Motivation of tSNE: Try to model the relative distances between data points in high dimension into probabilities.
Then, create new data points having the same probability.

tSNE is an improvement of SNE (2002)

tSNE require a huge amount of RAM for big dataset.

Result: the higher distance in high-dimensional data points, the higher distance in low-dimensional data points.
"""
import matplotlib.pyplot as plt
import utils
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def plot_3D(X, y):
    X_embedded = TSNE(n_components=3,  # convert into 3-D dimension
                      verbose=1,
                      perplexity=40,  # The perplexity is related to the number of nearest neighbors.
                      n_iter=300,  # the number of updating data point y_i
                      learning_rate=200  # the speed of updating data point y_i in low dimension
                      ).fit_transform(X, y)

    ax = plt.axes(projection='3d')
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y)
    plt.show()


def plot_2D(X, y):
    X_embedded = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500).fit_transform(X, y)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, alpha=0.5)
    plt.show()


def main():
    X, y = utils.readCsv('./data/digit-recognizer/train.csv', limit=2000)
    plot_3D(X, y)


main()
