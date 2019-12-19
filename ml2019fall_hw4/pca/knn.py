import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    num_test = x.shape[0]
    num_train = x_train.shape[0]
    y = np.zeros(num_test)

    for i in range(num_test):

        dist = np.sqrt(np.sum((x[i] - x_train) ** 2, axis=1))
        idx = np.argsort(dist)[:k]
        # y[i] = np.argmax(np.bincount(y_train[idx]))
        y[i] = scipy.stats.mode(y_train[idx])[0]

    # end answer

    return y
