import numpy as np
import numpy.random as rgt
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_kernels as PK
from scipy.linalg import pinv, sqrtm
import scipy


def gen_data_valid(n = 2000, random_state = 2024):
        '''
        Data Generating Process in Cui_etal_2024
        '''
        rgt.seed(random_state)

        X1 = 0.25 + 0.25*rgt.normal(size = n)
        X2 = 0.25 + 0.25*rgt.normal(size = n)

        prob_A = expit(-(1/8)*X1 - (1/8)*X2)
        A = rgt.binomial(1, prob_A, n)

        cov_matrix = [[1, 1/4, 1/2],
                        [1/4, 1, 1/2],
                        [1/2, 1/2, 1]]

        err = rgt.multivariate_normal([0,0,0], cov_matrix, n)
        Z = 1/4 + 1/4*A + 1/4*X1 + 1/4*X2 + err[:,0].reshape(-1)
        W = 1/4 + 1/8*A + 1/4*X1 + 1/4*X2 + err[:,1].reshape(-1)
        U = 1/4 + 1/4*A + 1/4*X1 + 1/4*X2 + err[:,2].reshape(-1)    

        Y = 9/4 + 2*A + 1/2*X1 + 1/2*X2 + U + 2*W + 0.25*rgt.normal(size = n)

        return {'X': np.column_stack((X1, X2)), 'A': A.reshape(-1,1), 'Z': Z.reshape(-1,1), 'W': W.reshape(-1,1), 'Y': Y.reshape(-1,1)}

def expit(x):
        # expit function
        return (1 + np.exp(-x))**(-1)