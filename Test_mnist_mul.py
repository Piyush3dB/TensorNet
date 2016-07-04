import scipy.io
import sys
import pdb as pdb
import numpy as np


def my_tt_mul(W, A):

    n  = W['n'][0,0]     # sizes of row indices
    m  = W['m'][0,0]     # sizes of col indices
    tt = W['tt'][0,0]    # TT-tensor of the vectorized TT-representation of the matrix
    

    cra= tt['core'] # cores of the TT-decomposition stored in one 'long' 1D array
    d  = tt['d']    # dimension of the array
    ps = tt['ps']   # markers for position of the k-the core in array tt.core
    r  = tt['r']    


    rb = np.shape(A)[1]      # number of batches
    ns = np.vstack((m, rb))        # new shape for input data
    c  = np.reshape(A, ns, order='F')   # reshape input data to new shape


    pdb.set_trace()
    



def main():
    mat = scipy.io.loadmat('./experiments/mnist/mnist.mat')

    W = mat['W']
    A = mat['A']


    m2 = my_tt_mul(W, A)




    #pdb.set_trace()



# Entry point
if __name__ == "__main__":
    sys.exit(main())
