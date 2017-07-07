###
###
###

# external packages
import numpy as np

# declare fused types for wavelet and covariate arrays
ctypedef fused waveType:
    float
    double

ctypedef fused covType:
    int
    float
    double

# global variables
cdef:
    float EPS = 1e-50       # buffer to prevent dividing by zero.
    float thresh = 1e-20    # convergence threshold for EM
    int niters = 1000

# functions for BayesTDA-Cython independence model
cdef expectationMaximization(int num_samples,
                             int num_kernels,
                             int num_wavelets,
                             waveType[:,:] wavelet_coeffs,
                             covType[:] covariates):

    cdef double lnLikelihood, oldLnLikelihood = 0.0
    cdef int iter = 0

    while True:
        iter += 1
        oldLnLikelihood = lnLikelihood

        # TODO: parallelize this step.
        for k in range(num_kernels):


#
