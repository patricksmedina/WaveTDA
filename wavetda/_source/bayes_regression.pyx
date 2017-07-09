###
###
###

# external packages
cimport numpy as cnp
import numpy as np
import os

# global variables
cdef:
    double EPS = 1e-50     # buffer to prevent errors with zero in division/log.
    double THRESH = 1e-20 # convergence threshold for EM
    int NITERS = 1000     # number of EM iterations

# bayes_regression independence model class
cdef class bayes_regression:

    # initialize variable types
    cdef:
        # constants needed throughout computations
        int num_samples, num_kernels, num_scales
        int num_wavelets, num_covariates, niter
        double lnLikelihood, ss_a
        str outdir

        # arrays
        float[:,:,:] wavelet_coeffs     # stores the wavelet coefficients
        float[:,:] pi_estimates         # stores the EM estimates
        float[:,:] covariates           # stores the covariate values
        float[:,:] lnBFs                # stores the logBFs
        long[:] use                     # stores the use array

    # Python class initialization
    def __init__(self, num_samples, num_kernels, num_scales, num_wavelets,
                 num_covariates, wavelet_coeffs, covariates, use, outdir):

        # store variables to self
        self.num_samples = num_samples
        self.num_kernels = num_kernels
        self.num_scales = num_scales
        self.num_wavelets = num_wavelets
        self.num_covariates = num_covariates
        self.outdir = outdir
        self.niter = 0
        self.lnLikelihood = 0
        self.ss_a = 0.05

        # store arrays to self
        self.wavelet_coeffs = wavelet_coeffs
        self.covariates = covariates
        self.use = use

        # initialize estimates of pi and lnBFs
        self.pi_estimates = np.ones((num_kernels, (num_scales + 1)), dtype=np.float32) / (num_scales + 1)
        self.lnBFs = np.zeros((num_kernels, num_wavelets), dtype=np.float32)

    def __call__(self):
        """
        Executes EM algorithm.
        """

        self.expectationMaximization()

    def _loadLnBFs(self, cov):
        """
        Load log Bayes Factors of a set of persistence kernels for a given covariate.
        """

        fname = os.path.join(self.outdir,"lnBFs","cov_{}.csv".format(cov))
        temp_csv = np.genfromtxt(fname, delimiter=",")
        self.logBFs = temp_csv.astype(np.float32)

    def _savelnBFs(self, cov):
        """
        Save log Bayes Factors of a set of persistence kernels for a given covariate.
        """

        fname = os.path.join(self.outdir,"lnBFs","cov_{}.csv".format(cov))
        np.savetxt(fname, np.asarray(self.lnBFs), delimiter=",")
        self.lnBFs = np.zeros((self.num_kernels, self.num_wavelets), dtype=np.float32)

    # functions for BayesTDA-Cython independence model
    cdef void computeLnBayesFactor(self, int ker, int wav, int cov):
        """
        Compute the log Bayes Factors used in the EM algorithm.
        """

        # extract needed arrays
        w = np.asarray(self.wavelet_coeffs[ker, :, wav])
        x = np.asarray(self.covariates[:, cov])

        # sum of squares
        sww = np.dot(w,w)
        swx = np.array([w.sum(), np.dot(w,x)], dtype=np.float32).reshape(1,2)
        omega = np.array([[self.num_samples, x.sum()],[x.sum(), np.dot(x,x) + self.ss_a]], dtype=np.float32)

        # numerator / denominator
        num = np.log(sww - self.num_samples * np.mean(w) ** 2 + EPS)
        denom = np.log(sww - np.dot(np.dot(swx, np.linalg.inv(omega)), swx.T) + EPS)

        # compute bayes factor
        tempBF = -np.log(self.ss_a) + 0.5 * np.log(self.num_samples)
        tempBF += 0.5 * np.log(np.linalg.det(omega) + EPS)
        tempBF += 0.5 * self.num_samples * (num - denom)

        self.lnBFs[ker, wav] = tempBF

    cdef double computeLnLikelihood(self, int ker, int wav, int cov, int length):
        """

        """

        if self.niter == 1:
            self.computeLnBayesFactor(ker, wav, cov)

        # computeLnLikelihood value and update temp_pi_estimate
        pass

    cdef void expectationMaximization(self):
        """

        """
        # declare internal variables
        cdef:
            double oldLnLikelihood
            double relative_likelihood

            # start and stop indices for the wavelets at a scale
            int start_idx, end_idx

        # main EM loop -- mimics do-while in C++
        while True:
            # update variables
            self.niter += 1
            oldLnLikelihood = self.lnLikelihood

            # loop across covariates -- TODO: Parallelize this
            for cov in range(self.num_covariates):

                # load lnBFs for covariate
                if self.niter > 1:
                    self._loadLnBFs(cov)

                # loop across the persistence kernels
                for ker in range(self.num_kernels):
                    # zero out range of indices
                    start_idx = 0
                    end_idx = 0

                    # loop across the scales
                    for sca in range(self.num_scales + 1):

                        # get indices of wavelet coefficients at that scale
                        if sca == 0:
                            start_idx = 0
                            stop_idx = 1
                        else:
                            start_idx += stop_idx
                            stop_idx += 3 * (4 ** (sca - 1))

                        # loop across the wavelet coefficients at that scale
                        for wav in range(start_idx, stop_idx):

                            if self.use[wav] == 1:
                                # get the number of scales
                                length = stop_idx - start_idx

                                # update the log-likelihood with the given value
                                self.lnLikelihood += self.computeLnLikelihood(ker, wav, cov, length)
                            # END IF
                        # END WAVELET LOOP
                    # END SCALE LOOP
                # END KERNEL LOOP

                # save lnBFs for covariate
                if self.niter == 1:
                    self._savelnBFs(cov)

            # END COVARIATE LOOP

            # compute the relative likelihood and check if convergence criteria is met
            relative_likelihood = abs(self.lnLikelihood - oldLnLikelihood) / abs(oldLnLikelihood + EPS)

            if relative_likelihood < THRESH or self.niter > NITERS:
                break

        # END WHILE
# END CLASS
