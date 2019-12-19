import numpy as np
import matplotlib.pyplot as plt
from pca import PCA 
from PIL import Image
from math import acos
from scipy import misc

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer

    coord = np.where(img_r[:, :, -1] > 0)
    coord = np.array(coord).T
    
    eigvec, eigval = PCA(coord)
    angle = acos(eigvec[0,0] / np.linalg.norm(eigvec[:,0])) / np.pi * 180 - 90

    return misc.imrotate(img_r, angle)

    # end answer