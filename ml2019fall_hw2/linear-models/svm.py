import numpy as np
import scipy.optimize as optim

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    def fun(w):
        return 0.5 * np.matmul(w.T, w)

    X = np.vstack((np.ones((1, N)), X))

    cons = ({'type': 'ineq', 
             'fun': lambda w:  (y * np.matmul(w.T, X) - np.ones((1, N))).reshape(N)})
    res = optim.minimize(fun=fun, 
                         x0=w, 
                         constraints = cons, 
                         method = 'SLSQP')

    w = res.x
    dist = y * np.matmul(w.T, X)
    num = np.sum(abs(dist-1) < 0.0001)

    # end answer
    return w, num

