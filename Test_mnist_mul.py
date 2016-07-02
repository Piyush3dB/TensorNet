import scipy.io
import pdb as pdb
import numpy as np


def my_tt_mul(W, A):

    pass



def main():
    mat = scipy.io.loadmat('./experiments/mnist/mnist.mat')

    W = mat['W']
    A = mat['A']


    m2 = my_tt_mul(W, A)




    #pdb.set_trace()



# Entry point
if __name__ == "__main__":
    sys.exit(main())
