import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    
    for i in range(N):
        for j in range(K):
            x, mu, sigma = X[:, i], Mu[:, j], Sigma[:, :, j]
            p[i, j] = Phi[j] / (2 * np.pi * np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * (x-mu).dot(np.linalg.inv(sigma).dot(x-mu).T))
    p = p / np.sum(p, axis=1, keepdims=True)

    # end answer
    
    return p
    