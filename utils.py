import numpy as np
import pandas as pd


def softmax(X):
    X = np.exp(X)
    sum = np.sum(X, axis=1, keepdims=True)
    X = X / sum
    return X


def convert2indicator(y):
    '''
    Convert category into indicator matrix
    :param y:
    :return:
    '''
    max_value = np.amax(y) + 1
    Y = np.zeros((len(y), max_value))
    for idx, value in enumerate(y):
        Y[idx, value] = 1
    return Y


def readCsv(pathStr, limit=None):
    data = pd.read_csv(pathStr)
    data = data.to_numpy()  # convert to ndarray, headers removed
    # np.random.shuffle(data)
    X = data[:limit, 1:] * 1.0 / 255  # convert to the range of [0..1] (normalization step)
    y = data[:limit, 0]  # labels
    print('Load data from ' + str(pathStr) + ' done')
    return X, y
