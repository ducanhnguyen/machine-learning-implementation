# Implementation of principle component analysis (PCA)
# Input: matrix X (NxD), where N is the number of observations, D is the number of features
# Output: matrix Z (NxD)
# Problem: Find a matrix Q (DxD) such that Z = XQ
#
# Some properties of Z:
# - The variance of features decreases in sequential order
# (i.e., the first feature has the largest variance, the last feature has the least variance)
# - We can remove some right most columns of Z which are considered as noise (e.g., only keep 95% number of Z's columns)
#
# Usage of Z:
# - Dimensionality reduction: remove some right most columns
# - Decorrelation: In X, there exist correlations between two different features (see covariance matrix K_XX)
# . In Z, there does not exist any correlation between two different features.
# It means that all features in Z have the same importance in model construction.
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import utils


class PCA:
    def __init__(self, X):
        """

        :param X: The original matrix
        """
        self.X = X

    def transform(self):
        """
        Linear transformation
        :return: the transformation matrix Z
        """
        covMatrix = np.cov(
            m=self.X,  # shape: (NxD)
            rowvar=False  # each row represents an observation, with features in the columns.
        )

        # eigenvalues: 1-D array
        # eigenvectors: square 2-D array in which each column is an eigenvector
        eigenvalues, eigenvectors = LA.eigh(
            a=covMatrix  # shape (D, D)
        )

        # Sort eigenvalues in descending order
        sortedIndex = np.flip(np.argsort(eigenvalues))  # shape (D, 1)

        # Sort eigenvectors in the corresponding order
        sortedEigenvectors = eigenvectors[:, sortedIndex]  # shape (D, D)

        Z = self.X.dot(sortedEigenvectors)

        # just for testing
        # plot variances of each future
        variances = []
        for i in range(0, Z.shape[1]):
            variances.append(np.var(Z[:, i]))

        plt.plot(variances)
        plt.title("Variance of each component")
        plt.show()

        # cumulative variance
        plt.plot(np.cumsum(variances))
        plt.title("Cumulative variance")
        plt.show()
        return Z


def main():
    np.set_printoptions(threshold=sys.maxsize)

    X, Y = utils.readCsv('./data/digit-recognizer/train.csv', limit=2000)
    pca = PCA(X)  # do not apply PCA on labels
    pca.transform()

main()
