# helper functions

import numpy as np

# log of determinant of a matrix
# Cholesky decomposition is twice as fast as LU-decomposition when used to calculate general matrix determinants
def logdet(A):

    U = np.linalg.cholesky(A).T      # A = (U.T) * U
    y = 2*np.sum(np.log(np.diag(U))) # logdet(A) = log(det(A)) where det(A) = product of squares of diagonal elements

    return y

# TODO Trench method
# Invert a symmetric, real, positive definite Toeplitz matrix
# using inv() or Trench algorithm
def invToeplitz(T):

    Ti = np.linalg.inv(T) # TODO can change to OLS
    ld = logdet(T)

    return (Ti,ld)

# TODO
# Inverts a matrix that is block persymmetric
def invPerSymm(M, blk_size, off_diag_sparse):
    
    invM = np.linalg.inv(M) # TODO use block persymmetric property
    logdet_M = -logdet(invM)

    return invM, logdet_M

# TODO
# Fills in the bottom half of a block persymmetric matrix, given the top half
def fillPerSymm(Pin, blk_size, T, usr_blk_size_vert = None):

    blk_size_vert = blk_size
    if usr_blk_size_vert is not None:
        blk_size_vert = usr_blk_size_vert

    # Fill in bottom half
    T_half = int(np.floor(T/2.))
    idx_half = np.arange(0,blk_size_vert).reshape((blk_size_vert,1)) \
                + np.arange((T_half-1),-1,-1)*blk_size_vert
    idx_full = np.arange(0,blk_size).reshape((blk_size,1)) \
                + np.arange((T-1),-1,-1)*blk_size

    Pout = np.concatenate([Pin, Pin[np.ix_(idx_half.flatten(order="F"), idx_full.flatten(order="F"))]], 0)
    # use ‘F’ to flatten in column-major (Fortran- style) order.

    return Pout
