import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer

    data = (data - np.mean(data, axis=0, keepdims=True)) # / np.std(data, axis=0, keepdims=True) 
    eigval, eigvec = np.linalg.eig(np.cov(data.T))

    sorted_eig_idx = np.argsort(-eigval)
    eigvalue = eigval[sorted_eig_idx]
    eigvector = eigvec[:, sorted_eig_idx]

    return eigvector, eigvalue

    # end answer
