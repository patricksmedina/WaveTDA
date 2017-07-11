###
###
###

# external packages
cimport numpy as cnp
import numpy as np
import os

# global variables
cdef:
    double EPS = 1e-200                  # buffer to prevent errors with zero in division/log.
    double THRESH = 1e-20               # convergence threshold for EM
    int NITERS = 1000                   # number of allowable EM iterations

# bayes_regression independence model class
cdef class bayes_regression:

    # initialize variable types
    cdef:
        # Constants
        int num_samples                 # number of samples
        int num_kernels                 # number of persistence kernels
        int num_scales                  # number of scales used for kernels
        int num_wavelets                # total number of wavelets for kernels
        int num_covariates              # number of covariates
        int niter                       # current iteration number in the EM algorithm
        double lnLikelihood             # log Likelihood value
        double ss_a                     # prior \sigma^{2}_{a}
        str outdir                      # directory where files are saved

        # Arrays
        double[:,:,:] wavelet_coeffs    # stores the wavelet coefficients
        float[:,:,:] parameters         # stores the parameters for the posterior analysis
        double[:,:] pi_estimates        # stores the EM estimates
        double[:,:] temp_pi_estimates   # stores the temporary pi estimates
        float[:,:] covariates           # stores the covariate values
        float[:,:] lnBFs                # stores the log Bayes Factors
        long[:] use                     # stores the use array

    # Python class initialization
    def __init__(self, num_samples, num_kernels, num_scales, num_wavelets,
                 num_covariates, wavelet_coeffs, covariates, use, outdir,
                 ss_a = 0.4):

        # store variables to self
        self.num_samples = num_samples
        self.num_kernels = num_kernels
        self.num_scales = num_scales
        self.num_wavelets = num_wavelets
        self.num_covariates = num_covariates
        self.outdir = outdir
        self.ss_a = ss_a
        self.niter = 0
        self.lnLikelihood = 0.0

        # store arrays to self
        self.wavelet_coeffs = wavelet_coeffs
        self.covariates = covariates
        self.use = use

        # initialize estimates of pi and lnBFs
        self.temp_pi_estimates = np.zeros((num_kernels, num_scales + 1), dtype=np.float64)
        self.pi_estimates = 0.1 * np.ones((num_kernels, num_scales + 1), dtype=np.float64) / (num_scales + 1)
        self.lnBFs = np.zeros((num_kernels, num_wavelets), dtype=np.float32)
        self.parameters = np.zeros((num_kernels, 2, num_wavelets), dtype=np.float32)

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
        temp_csv = temp_csv.reshape(self.num_kernels, self.num_wavelets)
        self.lnBFs = temp_csv.astype(np.float32)

    def _savelnBFs(self, cov):
        """
        Save log Bayes Factors of a set of persistence kernels for a given covariate.
        """

        fname = os.path.join(self.outdir,"lnBFs","cov_{}.csv".format(cov))
        np.savetxt(fname, np.asarray(self.lnBFs).reshape(self.num_kernels, self.num_wavelets), delimiter=",")
        self.lnBFs = np.zeros((self.num_kernels, self.num_wavelets), dtype=np.float32)

    def _loadPiEstimates(self, cov):
        """
        Load pi estimates for use in the EM algorithm for the current step.
        """

        fname = os.path.join(self.outdir,"piEstimates","cov_{}.csv".format(cov))
        temp_pi = np.genfromtxt(fname, delimiter=",")
        temp_pi = temp_pi.reshape(self.num_kernels, self.num_scales + 1)

        self.pi_estimates = temp_pi.astype(np.float64)

    def _savePiEstimates(self, cov):
        """
        Save pi Estimates from EM algorithm for current step.
        """

        # generate the filename, generate the array
        # to save with persistence kernels in the first row and save the array
        fname = os.path.join(self.outdir,"piEstimates","cov_{}.csv".format(cov))
        np.savetxt(fname, np.asarray(self.pi_estimates).reshape(self.num_kernels, self.num_scales + 1), delimiter=",")

        # resets the pi_estimate array for the next covariate -- reset to default array if in first iteration
        if self.niter > 1:
            self.pi_estimates = np.zeros((self.num_kernels, self.num_scales + 1), dtype=np.float64)
        else:
            self.pi_estimates = 0.1 * np.ones((self.num_kernels, self.num_scales + 1), dtype=np.float64) / (self.num_scales + 1)

    def _saveParameters(self, cov):
        """
        Saves the parameters for the posterior analysis.
        """

        for ker in range(self.num_kernels):
            fname = os.path.join(self.outdir,"parameters","cov_{0}_ker_{1}.csv".format(cov, ker))
            np.savetxt(fname, np.asarray(self.parameters[ker,:,:]), delimiter=",")

        self.parameters = np.zeros((self.num_kernels, 2, self.num_wavelets), dtype=np.float32)

    def getOutdir(self):
        print("[INFO] Directory of output: {}".format(self.outdir))

    def getLnLikelihoodRatio(self):
        """
        Prints the log likelihood ratio stored to the object.  This function
        only makes sense to call after running the EM algorithm.
        """
        print("[INFO] Log likelihood ratio {}".format(self.lnLikelihood))

    cdef void computeLnBayesFactor(self, int ker, int wav, int cov):
        """
        Compute the log Bayes Factors used in the EM algorithm.
        """

        cdef double tempBF, num, denom, sww

        # extract needed arrays
        w = np.asarray(self.wavelet_coeffs[ker, :, wav])
        x = np.asarray(self.covariates[:, cov])

        # sum of squares
        sww = np.dot(w,w)
        swx = np.array([w.sum(), np.dot(w,x)], dtype=np.float32).reshape(1,2)
        omega = np.array([[self.num_samples, x.sum()],[x.sum(), np.dot(x,x) + (1.0 / self.ss_a)]], dtype=np.float32)

        # numerator / denominator
        num = np.log(sww - self.num_samples * (np.mean(w) ** 2) + EPS)
        denom = np.log(sww - np.dot(np.dot(swx, np.linalg.inv(omega)), swx.T) + EPS)

        # compute bayes factor
        tempBF = -np.log(self.ss_a) + 0.5 * np.log(self.num_samples)
        tempBF -= 0.5 * np.log(np.linalg.det(omega))
        tempBF += 0.5 * self.num_samples * (num - denom)

        self.lnBFs[ker, wav] = tempBF

        # compute parameters for posterior analysis
        # mean then scaling
        self.parameters[ker,0,wav] = np.dot(np.linalg.inv(omega), swx.T)[1,0]
        self.parameters[ker,1,wav] = (omega[0,0] / (omega[0][0] * omega[1][1] - omega[0][1] * omega[1][0]))
        self.parameters[ker,1,wav] = self.parameters[ker,1,wav] * (sww - np.dot(np.dot(swx, np.linalg.inv(omega)), swx.T))
        self.parameters[ker,1,wav] = self.parameters[ker,1,wav] / self.num_samples

    cdef double computeLnLikelihood(self, int ker, int wav, int cov, int sca):
        """
        Computes the log likelihood for a given persistence kernel,
        wavelet coefficient, and covariate.
        """

        cdef double gamma, ll, p

        if self.niter == 1:
            self.computeLnBayesFactor(ker, wav, cov)

        # computeLnLikelihood value and update temp_pi_estimates
        p = self.pi_estimates[ker, sca]
        gamma = p * np.exp(self.lnBFs[ker, wav])
        gamma = gamma / (1 - p + gamma)

        # update the temporary pi estimate
        self.temp_pi_estimates[ker, sca] = self.temp_pi_estimates[ker, sca] + gamma

        # compute the lnLikelihood
        ll = gamma * (np.log(p + EPS) + self.lnBFs[ker, wav] - np.log(1 - p + EPS))
        ll = ll + np.log(1 - p + EPS)

        return(ll)

    cdef void expectationMaximization(self):
        """
        Performs the EM algorithm on the given input.
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

                # load lnBFs and previous piEstimates for covariate
                if self.niter > 1:
                    self._loadLnBFs(cov)
                    self._loadPiEstimates(cov)
                    self.parameters = None

                # loop across the scales
                for sca in range(self.num_scales + 1):

                    # get indices of wavelet coefficients at that scale
                    if sca == 0:
                        start_idx = 0
                        stop_idx = 1
                    else:
                        start_idx = stop_idx
                        stop_idx += 3 * (4 ** (sca - 1))

                    # loop across the persistence kernels
                    for ker in range(self.num_kernels):

                        # loop across the wavelet coefficients at that scale
                        for wav in range(start_idx, stop_idx):

                            if self.use[wav] == 1:
                                # update the log-likelihood with the given value
                                self.lnLikelihood += self.computeLnLikelihood(ker, wav, cov, sca)
                            # END IF

                        # END WAVELET LOOP
                    # END KERNEL LOOP

                    # normalize the temp_pi_estimates
                    for ker in range(self.num_kernels):
                        self.temp_pi_estimates[ker, sca] = self.temp_pi_estimates[ker,sca] / (stop_idx - start_idx)
                # END SCALE LOOP

                # overwrite the pi_estimates at this scale and zero out the temp_pi_estimates array
                self.pi_estimates = self.temp_pi_estimates
                self.temp_pi_estimates = np.zeros((self.num_kernels, self.num_scales + 1), dtype=np.float64)

                # save the pi_estimates for this covariate
                self._savePiEstimates(cov)

                # save lnBFs for covariate
                if self.niter == 1:
                    self._savelnBFs(cov)
            # END COVARIATE LOOP

            # save parameters for covariate
            if self.niter == 1:
                self._saveParameters(cov)

            # compute the relative likelihood and check if convergence criteria is met
            relative_likelihood = abs(self.lnLikelihood - oldLnLikelihood) / abs(oldLnLikelihood + EPS)
            if relative_likelihood < THRESH or self.niter > NITERS:
                print("[INFO] Method converged in {} iterations".format(self.niter))
                break

        # END WHILE
# END CLASS
