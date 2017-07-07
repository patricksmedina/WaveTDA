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
cdef float computeLnBayesFactor(int num_samples,
                                waveType[:] wavelet_coeffs,
                                covType[:] covariates):
    pass

cdef expectationMaximization(int num_samples,
                             int num_kernels,
                             int num_wavelets,
                             waveType[:,:] wavelet_coeffs,
                             covType[:] covariates,
                             int[:] use):

    cdef:
        double lnLikelihood, oldLnLikelihood = 0.0
        int iter = 0

    # main EM loop
    while True:
        iter += 1
        oldLnLikelihood = lnLikelihood

        for k in range(num_kernels):


#
