import scipy.io
import sys
import pdb as pdb
import numpy as np


def my_tt_mul_F(W, A):

    n  = W['n'][0,0]     # sizes of row indices
    m  = W['m'][0,0]     # sizes of col indices
    tt = W['tt'][0,0]    # TT-tensor of the vectorized TT-representation of the matrix
    
    # Weight
    cra= tt['core'][0,0] # cores of the TT-decomposition stored in one 'long' 1D array
    d  = tt['d'][0,0]    # dimension of the array
    ps = tt['ps'][0,0]-1   # markers for position of the k-the core in array tt.core
    r  = tt['r'][0,0]    # ranks of the decomposition

    # Data
    rb = np.shape(A)[1]      # number of batches
    ns = np.vstack((m, rb))        # new shape for input data
    c  = np.reshape(A, ns, order='F')   # reshape input data to new shape



    for k in range(d[0,0]):

        
        # Extract core
        cr = cra[ps[k]:ps[k+1], :]

        # Create core shape
                       #   0 1 2 3
        cs = np.hstack((r[k], n[k], m[k], r[k+1]))
        print 'Core shape ' + str((cs))

        # Reshape core
        cr = np.reshape(cr, cs, order='F')

        # Swap axes
        cr = cr.transpose((1,3,0,2))

        # Reshape to matrix -> 'weight'
                           # 1    3       0    2
        cr = np.reshape(cr, (n[k]*r[k+1], r[k]*m[k]), order='F')
        #pdb.set_trace()
        
        print 'Core shape ' + str(np.shape(cr))


        M = np.size(c)
        c = np.reshape(c, (r[k]*m[k], M/(r[k]*m[k])), order='F')
        
        print 'C shape ' + str(np.shape(c))

        c = np.einsum('ij,jk->ik', cr, c)
        c = np.reshape(c, (n[k], np.size(c)/n[k]), order='F')
        c = c.transpose((1,0))

        print ' '

    c = np.reshape(c, (np.size(c), 1), order='F')
    c = np.reshape(c, (rb, np.size(c)/rb), order='F')
    c = c.transpose((1,0))

    return c

    
        


def my_tt_mul_C(W, A):

    n  = W['n'][0,0]     # sizes of row indices
    m  = W['m'][0,0]     # sizes of col indices
    tt = W['tt'][0,0]    # TT-tensor of the vectorized TT-representation of the matrix
    
    # Weight
    cra= tt['core'][0,0] # cores of the TT-decomposition stored in one 'long' 1D array
    d  = tt['d'][0,0]    # dimension of the array
    ps = tt['ps'][0,0]-1   # markers for position of the k-the core in array tt.core
    r  = tt['r'][0,0]    # ranks of the decomposition

    # Data
    rb = np.shape(A)[1]      # number of batches
    ns = np.vstack((m, rb))        # new shape for input data
    c  = np.reshape(A, ns, order='C')   # reshape input data to new shape



    for k in range(d[0,0]):

        
        # Extract core
        cr = cra[ps[k]:ps[k+1], :]

        # Create core shape
                       #   0 1 2 3
        cs = np.hstack((r[k], n[k], m[k], r[k+1]))
        print 'Core shape ' + str((cs))

        # Reshape core
        cr = np.reshape(cr, cs, order='C')

        # Swap axes
        cr = cr.transpose((1,3,0,2))

        # Reshape to matrix -> 'weight'
                           # 1    3       0    2
        cr = np.reshape(cr, (n[k]*r[k+1], r[k]*m[k]), order='C')
        #pdb.set_trace()
        
        print 'Core shape ' + str(np.shape(cr))


        M = np.size(c)
        c = np.reshape(c, (r[k]*m[k], M/(r[k]*m[k])), order='C')
        
        print 'C shape ' + str(np.shape(c))

        c = np.einsum('ij,jk->ik', cr, c)
        c = np.reshape(c, (n[k], np.size(c)/n[k]), order='C')
        c = c.transpose((1,0))

        print ' '

    c = np.reshape(c, (np.size(c), 1), order='C')
    c = np.reshape(c, (rb, np.size(c)/rb), order='C')
    c = c.transpose((1,0))

    return c



    



def main():
    mat = scipy.io.loadmat('./experiments/mnist/mnist.mat')

    W = mat['W']
    A = mat['A']
    m = mat['m']


    m2 = my_tt_mul_F(W, A)
    m3 = my_tt_mul_C(W, A)

    print np.sum(m-m2)

    pdb.set_trace()

    #mt = m.transpose((1,0))
    
    print np.sum(m-m3)







# Entry point
if __name__ == "__main__":
    sys.exit(main())
