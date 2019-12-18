import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer

    max_iter = 2000
    learning_rate = 0.001

    X = np.vstack((np.ones((1, N)), X))
    while(iters < max_iter):
        f = np.matmul(w.T, X)
        y_pred = np.sign(f)
        
        # update: w_t = w_(t-1) + xy
        # criterion: minimize(wxy)
        grad = np.matmul(X, 0.5 * (y - y_pred).T) / N 
        if np.sum(grad) == 0:
            break
        w += learning_rate * grad
        iters += 1

    # end answer
    
    return w, iters