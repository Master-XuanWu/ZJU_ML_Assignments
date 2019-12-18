import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''

    P, N = X.shape
    # w = np.zeros((P + 1, 1))
    w = np.random.rand(P+1, 1)
    iters = 0
    # YOUR CODE HERE

    # begin answer

    max_iter = 3000
    learning_rate = 0.01

    X = np.vstack((np.ones((1, N)), X))
    while(iters < max_iter):
        f = np.matmul(w.T, X)
        y_pred = 1 / (1 + np.exp(-f))
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5] = 0 

        if np.sum(y_pred==y) == N:
            break        
        grad = (np.matmul(X, 0.5 * (y - y_pred).T) + lmbda * w ) / N
        w += learning_rate * grad
        # learning_rate = np.power(0.9, iters//100)
        iters += 1
    
    return w
