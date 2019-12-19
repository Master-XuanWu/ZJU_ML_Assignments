import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer

    n, p = X.shape
    W = np.identity(n) * threshold
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(X[i, :] - X[j, :])
            W[i, j] = distance
            W[j, i] = distance
    W = np.where(W < threshold, 1, 0)
    return W

    # end answer
