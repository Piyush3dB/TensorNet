import scipy.io
import sys
import pdb as pdb
import numpy as np


def my_tt_mul_F(W, A):


    n  = W[0,0]['n'][0,0]     # sizes of row indices. Output Modes.
    m  = W[0,0]['m'][0,0]     # sizes of col indices. Input modes.
    tt = W[0,0]['tt'][0,0]    # TT-tensor of the vectorized TT-representation of the matrix
    

    # Weight
    cra    = tt['core'][0,0]   # cores of the TT-decomposition stored in one 'long' 1D array
    nCores = tt['d'][0,0][0,0] # dimension of the array
    ps     = tt['ps'][0,0]-1   # markers for position of the k-the core in array tt.core
    r      = tt['r'][0,0]      # ranks of the decomposition


    # Data
    rb = np.shape(A)[1]      # number of batches
    ns = np.vstack((m, rb))        # new shape for input data
    #pdb.set_trace()
    c  = np.reshape(A, ns, order='F')   # reshape input data to new shape
    # c is [4 x 8 x 8 x 4 x 100]


    # For every core in te TT
    for k in range(nCores):

        # Extract core
        core = cra[ps[k]:ps[k+1], :]

        # Create core shape
                       #   0 1 2 3
        cs = np.hstack((r[k], n[k], m[k], r[k+1]))

        # Reshape core
        core = np.reshape(core, cs, order='F')

        # Swap axes
        core = core.transpose((1,3,0,2))

        # Reshape to matrix -> 'weight'
                           # 1    3       0    2
        core = np.reshape(core, (n[k]*r[k+1], r[k]*m[k]), order='F')

        print ('Core %1d. Size = %4d, Shape = %s, Shape = %s') % (k, np.size(core), str(cs), str(np.shape(core)))

        M = np.size(c)
        c = np.reshape(c, (r[k]*m[k], M/(r[k]*m[k])), order='F')
        
        print ('  Data. Size = %7d. Shape = %s ') % ( M, str(np.shape(c)))

        c1 = np.einsum('ij,jk->ik', core, c)
        c2 = np.reshape(c1, (n[k], np.size(c1)/n[k]), order='F')
        c = c2.transpose((1,0))

        print ('  Result.  Size = %7d. Shape = %s. Shape = %s. Shape = %s ') % (np.size(c1), str(np.shape(c1)), str(np.shape(c2)), str(np.shape(c)))

        print ' '

    print 'Final reshape'
    c = np.reshape(c, (np.size(c), 1), order='F')
    c = np.reshape(c, (rb, np.size(c)/rb), order='F')
    c = c.transpose((1,0))

    print ('C Shape = %s') % (str(np.shape(c)))

    #pdb.set_trace()

    return c

    
        

def vl_nntt_forward(layer, iin, out):
    """
    """

    inHeight   = np.shape(iin['x'][0,0])[0]
    inWidth    = np.shape(iin['x'][0,0])[1]
    inChannels = np.shape(iin['x'][0,0])[2] if np.size(np.shape(iin['x'][0,0]))==4 else 1
    batchSize  = np.shape(iin['x'][0,0])[3] if np.size(np.shape(iin['x'][0,0]))==4 else 1




    # Weights
    W = layer['W'] # Weights

    pdb.set_trace()
    # Data
    A = np.reshape(iin['x'][0,0], (inHeight*inWidth, batchSize), order='F')
    

    # Call function
    m2 = my_tt_mul_F(W, A)

    return m2


    #return out


    



def main():

    #mat = scipy.io.loadmat('./experiments/mnist/mnist_1_batch_fwd.mat')
    mat = scipy.io.loadmat('./experiments/mnist/mnist_100_batch_fwd.mat')

    layer = mat['layer']
    iin   = mat['in']
    out   = mat['out']
    
    m     = mat['m']

    # Pass to function here
    m2 = vl_nntt_forward(layer, iin, out)

    
    print ('Result:: Matlab=%f  Py=%f  Difference = %f') % (float(np.sum(m)), float(np.sum(m2)), float(np.sum(m-m2)))

    #pdb.set_trace()






# Entry point
if __name__ == "__main__":
    sys.exit(main())
