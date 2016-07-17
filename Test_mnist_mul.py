import scipy.io
import sys
import pdb as pdb
import random
import numpy as np
import os
from numpy.testing import assert_allclose
sys.path.append("../mxnet/python")
import mxnet as mx
#from check_utils import (check_numeric_gradient, check_symbolic_backward, check_symbolic_forward, reldiff, _np_reduce)


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
    # c is [4 x 8 x 8 x 4 x batchSize]

    #pdb.set_trace()


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
        sc = np.shape(c)
        c = np.reshape(c, (r[k]*m[k], M/(r[k]*m[k])), order='F')
        
        print ('  Data. Size = %7d.  Shape = %s     Shape = %s ') % ( M, str(sc), str(np.shape(c)))

        c1 = np.einsum('ij,jk->ik', core, c)
        c2 = np.reshape(c1, (n[k], np.size(c1)/n[k]), order='F')
        c = c2.transpose((1,0))

        print ('\n  Result.  Size = %7d. Shape = %s. Shape = %s. Shape = %s ') % (np.size(c1), str(np.shape(c1)), str(np.shape(c2)), str(np.shape(c)))

        print ' '

    print 'Final reshape'
    c = np.reshape(c, (np.size(c), 1), order='F')
    c = np.reshape(c, (rb, np.size(c)/rb), order='F')
    c = c.transpose((1,0))

    print ('C Shape = %s') % (str(np.shape(c)))

    #pdb.set_trace()

    return c

    

def my_tt_mul_C(W, A):


    Om  = W[0,0]['n'][0,0]     # sizes of row indices. Output Modes.
    Im  = W[0,0]['m'][0,0]     # sizes of col indices. Input modes.
    tt  = W[0,0]['tt'][0,0]    # TT-tensor of the vectorized TT-representation of the matrix
    

    # Weight
    coreAry = tt['core'][0,0]   # cores of the TT-decomposition stored in one 'long' 1D array
    nCores  = tt['d'][0,0][0,0] # dimension of the array
    r       = tt['r'][0,0]      # ranks of the decomposition


    # Data
    rb = np.shape(A)[0]      # number of batches
    ns = np.vstack((Im, rb))        # new shape for input data
    
    ns = np.flipud(ns)

    c  = np.reshape(A, ns, order='C')

    pCrA = 0

    # For every core in te TT
    for k in range(nCores):


        # Create core shape
        cs  = np.hstack((r[k+1], Im[k], Om[k], r[k]))
        CrS = np.prod(cs)

        # Extract core
        core = coreAry[ pCrA:pCrA+CrS , :]
        pCrA = pCrA+CrS

        # Reshape core
        core = np.reshape(core, cs, order='C')
        
        # Swap axes
        core = core.transpose((1,3,0,2))
        
        # Reshape to matrix -> 'weight'
        core = np.reshape(core, (r[k]*Im[k], Om[k]*r[k+1]), order='C')

        print ('Core %1d. Size = %4d, Shape = %s, Shape = %s') % (k, np.size(core), str(cs), str(np.shape(core)))

        M  = np.size(c)
        sc = np.shape(c)

        c = np.reshape(c, (-1, r[k]*Im[k]), order='C')
        
        print ('  Data. Size = %7d.  Shape = %s     Shape = %s ') % ( M, str(sc), str(np.shape(c)))


        c1 = np.einsum('ij,jk->ik', c, core)

        c = np.reshape(c1, (-1, Om[k]), order='C')

        c = c.transpose()

        print ('\n  Result.  Size = %7d. Shape = %s. Shape = %s. Shape = %s ') % (np.size(c1), str(np.shape(c1)), str(np.shape(c)), str(np.shape(c)))

        print ' '

    print 'Final reshape'


    c = np.reshape(c, (1, np.size(c)), order='C')

    c = np.reshape(c, (rb, -1), order='C')
    
    

    print ('C Shape = %s') % (str(np.shape(c)))

    #pdb.set_trace()

    return c




 

def my_tt_mul_MX(W, A):



    Om  = W[0,0]['n'][0,0]     # sizes of row indices. Output Modes.
    Im  = W[0,0]['m'][0,0]     # sizes of col indices. Input modes.
    tt  = W[0,0]['tt'][0,0]    # TT-tensor of the vectorized TT-representation of the matrix
    

    # Weight
    coreAry = tt['core'][0,0]   # cores of the TT-decomposition stored in one 'long' 1D array
    nCores  = tt['d'][0,0][0,0] # dimension of the array
    r       = tt['r'][0,0]      # ranks of the decomposition

    #pdb.set_trace()

    def genSym(W, A, Om, Im, nCores):

        coreAry = mx.sym.Variable('coreAry')
        As = mx.sym.Variable('A')

        #pdb.set_trace()

        rb = np.shape(A)[0]
        ns = np.vstack((Im, rb))
        ns = tuple(np.flipud(ns)[:,0])
        
        c = mx.sym.Reshape(As, shape=ns)


        pCrA = 0
        
        # For every core in te TT
        for k in range(nCores):
            # Create core shape
            cs  = np.hstack((r[k+1], Im[k], Om[k], r[k]))
            CrS = np.prod(cs)

            # Extract core
            core = mx.symbol.slice_axis(data=coreAry, axis=0, begin=int(pCrA), end=int(pCrA+CrS))
            pCrA = pCrA+CrS

            # Reshape core
            core = mx.sym.Reshape(core, shape=tuple(cs))
            
            core = mx.sym.transpose(core, axes=(1,3,0,2))

            cs = tuple(np.hstack([r[k]*Im[k], Om[k]*r[k+1]]))
            core = mx.sym.Reshape(core, shape=cs)

            
            #cs = tuple(np.hstack(M/(r[k]*Im[k]), r[k]*Im[k]))
            cs = tuple(np.hstack([-1, r[k]*Im[k]]))
            c = mx.sym.Reshape(c, shape=cs)

            c1 = mx.sym.dot(c, core)

            cs = tuple(np.hstack([ -1 , Om[k] ]))
            c = mx.sym.Reshape(c1, shape=cs)
            c = mx.sym.transpose(c, axes=(1,0))



        c = mx.sym.Reshape(c, shape=(1,-1))
        c = mx.sym.Reshape(c, shape=(rb, -1))


        return c

        #c = mx.sym.dot(a, b)
        #coreAry = 


    sym = genSym(W, A, Om, Im, nCores)

    #pdb.set_trace()

    
    inp_shapes = {'coreAry':(400,1), 'A':(1,1024)}
    #inp_shapes = {'A':(1,1024)}
    
    arg_shapes, out_shapes, aux_shapes = sym.infer_shape(**inp_shapes)
    print arg_shapes
    print out_shapes

    pdb.set_trace()



    return c



 

def vl_nntt_forward_C(layer, iin, out):
    """
    """

    # Parameters
    inHeight   = np.shape(iin['x'][0,0])[0]
    inWidth    = np.shape(iin['x'][0,0])[1]
    inChannels = np.shape(iin['x'][0,0])[2] if np.size(np.shape(iin['x'][0,0]))==4 else 1
    batchSize  = np.shape(iin['x'][0,0])[3] if np.size(np.shape(iin['x'][0,0]))==4 else 1

    # Weights
    W = layer['W'] # Weights

    B = np.shape(layer['weights'][0,0][0,1])


    # Data
    #pdb.set_trace()
    At = iin['x'][0,0].transpose()
    A  = np.reshape(At, (batchSize, inHeight*inWidth), order='C')

    # Call function
#    m2 = my_tt_mul_C(W, A)
    m2 = my_tt_mul_MX(W, A)
    #pdb.set_trace()

    # Transpose for result
    m2 = m2.transpose()


    return m2



def vl_nntt_forward(layer, iin, out):
    """
    """

    # Parameters
    inHeight   = np.shape(iin['x'][0,0])[0]
    inWidth    = np.shape(iin['x'][0,0])[1]
    inChannels = np.shape(iin['x'][0,0])[2] if np.size(np.shape(iin['x'][0,0]))==4 else 1
    batchSize  = np.shape(iin['x'][0,0])[3] if np.size(np.shape(iin['x'][0,0]))==4 else 1

    # Weights
    W = layer['W'] # Weights

    B = np.shape(layer['weights'][0,0][0,1])


    # Data
    A = np.reshape(iin['x'][0,0], (inHeight*inWidth, batchSize), order='F')

    # Call function
    m2 = my_tt_mul_F(W, A)

    #pdb.set_trace()

    return m2


def vl_nntt_forward_nb(layer, iin, out):
    """
    """

    # Parameters
    inHeight   = np.shape(iin['x'][0,0])[0]
    inWidth    = np.shape(iin['x'][0,0])[1]
    inChannels = np.shape(iin['x'][0,0])[2] if np.size(np.shape(iin['x'][0,0]))==4 else 1
    batchSize  = np.shape(iin['x'][0,0])[3] if np.size(np.shape(iin['x'][0,0]))==4 else 1

    # Weights
    W = layer['W'] # Weights

    B = np.shape(layer['weights'][0,0][0,1])


    # Data
    A   = np.reshape(iin['x'][0,0], (inHeight*inWidth, batchSize), order='F')

    # Call function
    m2 = my_tt_mul_F(W, A)

    Anb = np.tile(A, (1,100))

    m3 = my_tt_mul_F(W, Anb)


    
    pdb.set_trace()


    return m2




def main():
    """
    """

    mat = scipy.io.loadmat('./experiments/mnist/mnist_1_batch_fwd.mat')
    #mat = scipy.io.loadmat('./experiments/mnist/mnist_100_batch_fwd.mat')

    layer = mat['layer']
    iin   = mat['in']
    out   = mat['out']
    
    m     = mat['m']

    # Pass to function here
    #m2 = vl_nntt_forward(layer, iin, out)
    #m2 = vl_nntt_forward_nb(layer, iin, out)
    m2 = vl_nntt_forward_C(layer, iin, out)
    #m2 = vl_nntt_forward_MX(layer, iin, out)

    
    print ('Result:: Matlab=%f  Py=%f  Difference = %f') % (float(np.sum(m)), float(np.sum(m2)), float(np.sum(m-m2)))

    #pdb.set_trace()



def test_swapaxes():
    data = mx.symbol.Variable('data')
    shape = (2, 3, 4)
    data_tmp = np.ones(shape)
    data_tmp[0] = 1
    data_tmp[1] = 2
    arr_data = mx.nd.array(data_tmp)
    swap0 = mx.symbol.SwapAxis(data=data, dim1=0, dim2=2)
    swap = mx.symbol.SwapAxis(data=swap0, dim1=1, dim2=2)
    exe_c = swap.bind(mx.cpu(), args=[arr_data])
    exe_c.forward()
    out = exe_c.outputs[0].asnumpy()

    swap0_ = np.swapaxes(data_tmp, 0, 2)
    swap_ = np.swapaxes(swap0_, 1, 2)

    assert reldiff(out, swap_) < 1e-6


def test_reshape():

    def test_reshape_new(src_shape, shape_args, dst_shape):
        net = mx.sym.Variable("data")
        net = mx.sym.Reshape(net, shape=shape_args)
        js = net.tojson()
        net = mx.sym.load_json(js)
        _, output_shape, __ = net.infer_shape(data=src_shape)
        assert output_shape[0] == dst_shape, \
            'Src Shape = %s, Shape Arguments = %s, Dst Shape = %s, Output Shape = %s' \
            %(str(src_shape), str(shape_args), str(dst_shape), str(output_shape[0]))
        dat_npy = np.random.rand(*src_shape)
        grad_npy = np.random.rand(*dst_shape)
        exe = net.simple_bind(mx.cpu(), data=src_shape)
        exe.arg_dict['data'][:] = dat_npy
        exe.forward(is_train=True)
        assert np.square(exe.outputs[0].asnumpy() - dat_npy.reshape(dst_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Dst Shape = %s' %(str(src_shape),
                                                                     str(shape_args), str(dst_shape))
        exe.backward(out_grads=mx.nd.array(grad_npy))
        assert np.square(exe.grad_dict['data'].asnumpy() - grad_npy.reshape(src_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Dst Shape = %s' %(str(src_shape),
                                                                     str(shape_args), str(dst_shape))
    # Test new api (Using shape)
    test_cases = [[(2, 3, 5, 5), (0, -1), (2, 75)],
                  [(2, 3, 5, 5), (0, 0, -1), (2, 3, 25)],
                  [(5, 3, 4, 5), (0, -1, 0), (5, 15, 4)],
                  [(2, 3, 5, 4), (-1, 0, 0), (8, 3, 5),
                  [(2, 3, 4, 5), (3, -1, 0), (3, 10, 4)],
                  [(2, 3, 5, 5), (5, 3, 0, -1), (5, 3, 5, 2)],
                  [(2, 3, 5, 5), (0, 0, 0, 0), (2, 3, 5, 5)],
                  [(2, 4, 5, 3), (-1, 2, 2, 1), (30, 2, 2, 1)]]]
    for test_case in test_cases:
        test_reshape_new(test_case[0], test_case[1], test_case[2])
    # Test old api
    net = mx.sym.Variable("data")
    net = mx.sym.Reshape(net, target_shape=(2, 0))
    js = net.tojson()
    net = mx.sym.load_json(js)
    _, output_shape, __ = net.infer_shape(data=(2, 3, 5, 5))
    assert(output_shape[0] == (2, 75))




def test_transpose():
    for ndim in range(1, 6):
        for t in range(5):
            dims = list(np.random.randint(1, 10, size=ndim))
            axes = list(range(ndim))
            random.shuffle(axes)
            axes = tuple(axes)
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.transpose(x, axes=axes)
            assert_allclose(np.transpose(x.asnumpy(), axes=axes), y.asnumpy())

            y = mx.nd.transpose(x)
            assert_allclose(np.transpose(x.asnumpy()), y.asnumpy())




def test_slice_axis():
    for ndim in range(1, 2):
        shape = np.random.randint(1, 11, size=(ndim,))
        for t in range(ndim):
            d = shape[t]
            b = random.randint(0, d-1)
            e = random.randint(b+1, d)
            idx = []
            for i in range(ndim):
                idx.append(slice(0, shape[i]))
            idx[t] = slice(b, e)

            X = mx.symbol.Variable('X')
            x = mx.nd.array(np.random.normal(size=shape))
            Y = mx.symbol.slice_axis(data=X, axis=t, begin=b, end=e)

            xgrad = mx.nd.empty(x.shape)
            exec1 = Y.bind(mx.cpu(), args = [x], args_grad = {'X': xgrad})
            exec1.forward()
            y = exec1.outputs[0]

            pdb.set_trace()
            assert_allclose(x.asnumpy()[idx], y.asnumpy())
            exec1.backward([y])
            xx = x.asnumpy()
            xx[:] = 0.0
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx, xgrad.asnumpy())



# Entry point
if __name__ == "__main__":
    sys.exit(main())
    #sys.exit(test_slice_axis())
    #sys.exit(test_swapaxes())
    #sys.exit(test_reshape())
    #sys.exit(test_transpose())
