import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

"""
Implementation of AutoRec. 
Paper: http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
Data: https://www.kaggle.com/grouplens/movielens-20m-dataset#rating.csv
"""
class AutoRec:

    def compute_batch_range(self, N, batch, batch_sz):
        """
        Compute batch range
        :param N: the number of observation ratings
        :param batch: the index of batch
        :param batch_sz: batch's size
        :return:
        """
        upper = np.min([N, (batch + 1) * batch_sz])
        lower = batch * batch_sz
        return lower, upper

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss
        :param y_true: the true output
        :param y_pred: the predicted output
        :return: loss
        """
        mask = y_true > 0
        mask = tf.dtypes.cast(mask, dtype=np.float32)  # convert float64 to float32

        diff = y_true - y_pred
        loss = np.multiply(diff, diff)  # element-wise multiplication
        loss = np.multiply(loss, mask)  # ignore the missing value (having value of zero) in loss computation
        sum = np.sum(loss)
        return sum

    def train_generator(self, Xtrain, batch_sz):
        """
        Generate batch samples. Use in fit_generator() in Keras
        :param Xtrain: input matrix NxD
        :param batch_sz: batch's size
        :return: batch samples
        """
        while True:  # # loop indefinitely. important!
            N = Xtrain.shape[0]
            n_batches = int(np.ceil(N / batch_sz))

            for batch in range(n_batches):
                lower, upper = self.compute_batch_range(N, batch, batch_sz)
                inputs = Xtrain[lower:upper, :]
                targets = inputs
                yield inputs, targets

    def fit(self, Xtrain, Xtest, batch_sz=1000, epoch=10):
        Ntrain, D = Xtrain.shape
        print('User-movie matrix shape: ' + str(Xtrain.shape))

        # create layer
        model = Sequential()
        model.add(Dense(input_dim=(D), units=256, activation='relu'))
        model.add(Dense(units=D, activation='relu'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=self.compute_loss)
        history = model.fit_generator(generator=self.train_generator(Xtrain, batch_sz), epochs=epoch,
                                      steps_per_epoch=int(np.ceil(Ntrain / batch_sz)),
                                      validation_data=(Xtest, Xtest),
                                      validation_steps=int(np.ceil(Xtest.shape[0] / batch_sz)))

        # plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.show()


def main():
    X = scipy.sparse.load_npz('../data/movielens-20m-dataset/rating.npz').todense()  # column: movie, row: user
    X = np.array(X)  # mantipulate on matrix is extremely slow
    cutoff = np.math.floor(X.shape[0] * 0.8)
    Xtrain = X[:cutoff, :]
    Xtest = X[cutoff:, :]

    autorec = AutoRec()
    autorec.fit(Xtrain, Xtest)


main()
