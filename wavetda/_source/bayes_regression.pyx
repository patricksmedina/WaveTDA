###
###
###

# external packages
cimport cnumpy as cnp
import numpy as np

# declare fused types for wavelet and covariate arrays
ctypedef fused waveType:
    float
    double

ctypedef fused covType:
    int
    float
    double

# bayes_regression independence model class
cdef class bayes_regression:
    # global variables
    cdef:
        float EPS = 1e-50     # buffer to prevent errors with zero in division/log.
        double THRESH = 1e-20 # convergence threshold for EM
        int NITERS = 1000     # number of EM iterations

    # initialize variable types
    cdef:
        int num_samples, num_kernels, num_scales, num_wavelets, num_covariates
        waveType[:,:,:] wavelet_coeffs
        covType[:,:] covariates
        int[:] use
        float[:, :] pi_estimates

    # Python class initialization
    def __init__(self, num_samples, num_kernels, num_scales, num_wavelets,
                 num_covariates, wavelet_coeffs, covariates, use):

                 # store variables to self
                 self.num_samples = num_samples
                 self.num_kernels = num_kernels
                 self.num_scales = num_scales
                 self.num_wavelets = num_wavelets
                 self.num_covariates = num_covariates

                 # store arrays to self
                 self.wavelet_coeffs = wavelet_coeffs
                 self.covariates = covariates
                 self.use = use

                 # initialize arrays and store to self
                 self.pi_estimates = np.array([1.0 / (num_scales + 1)] * (num_scales + 1),
                                              dtype=float)

    def __call__():
        self.expectationMaximization()

    # functions for BayesTDA-Cython independence model
    cdef double computeLnLikelihood(self,
                                    waveType[:] wavelets,
                                    covType[:] covariates):
        pass

    def expectationMaximization(self):
        # declare internal variables
        cdef:
            double lnLikelihood, oldLnLikelihood = 0.0
            double relative_likelihood
            int niter = 0

            # start and stop indices for the wavelets at a scale
            int start_idx, end_idx

        # main EM loop -- mimics do-while in C++
        while True:
            # update variables
            niter += 1
            oldLnLikelihood = lnLikelihood

            # loop across the persistence kernels
            for k in range(self.num_kernels):
                # zero out range of indices
                start_idx, end_idx = 0

                # loop across the scales
                for s in range(self.num_scales):

                    # get indices of wavelet coefficients at that scale
                    if s == 0:
                        start_idx = 0
                        stop_idx = 1
                    else:
                        start_idx += stop_idx
                        stop_idx += 3 * (4 ** (s - 1))

                    # loop across the wavelet coefficients at that scale
                    for w in range(start_idx, stop_idx):
                        if self.use[w] == 1:
                            # lnLikelihood += self.computeLnLikelihood()


            # compute the relative likelihood and check if convergence criteria is met
            relative_likelihood = abs(lnLikelihood - oldLnLikelihood) / abs(oldLnLikelihood)
            if (relative_likelihood < THRESH) or (niter > NITERS):
                break

        # END WHILE
#
