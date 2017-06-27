# from wavetda import persistence_diagram as pd
# from wavetda import persistence_kernels as pk
# from wavetda import persistence_wavelets as pw
import bayes_regression as br

import numpy as np

class BayesianRegression(object):

    def __init__(self, wavelet_coefficients, covariate_matrix, num_samples,
                 num_wavelets, num_covariates, num_scales, use = None, group_scaling = False):

        # check that the matrices are numpy arrays.
        if type(wavelet_coefficients) is not np.ndarray:
            raise TypeError("wavelet_coefficients must be of type 'numpy.ndarray.'")

        if type(covariate_matrix) is not np.ndarray:
            raise TypeError("covariate_matrix must be of type 'numpy.ndarray.'")

        # check that they are matrices.
        if len(wavelet_coefficients.shape) != 2:
            raise RuntimeError("wavelet_coefficients is not a matrix.")

        if len(covariate_matrix.shape) != 2:
            raise RuntimeError("covariate_matrix is not a matrix.")

        self.wavelet_coefficients = wavelet_coefficients
        self.covariate_matrix = covariate_matrix
        self.num_samples = num_samples
        self.num_wavelets = num_wavelets
        self.num_scales = num_scales
        self.num_covariates = num_covariates
        self.group_scaling = group_scaling
        self.use = use

        self.posterior_probs = None
        self.class_probs = None

    def __call__(self):
        """

        """
        if self.use is None:
            self.use = np.array((np.sum(self.wavelet_coefficients, axis=1) > 0) * 1)

        execute_br = br.BayesRegressionDM(self.num_scales,
                                          self.num_wavelets,
                                          self.num_covariates,
                                          self.num_samples,
                                          self.wavelet_coefficients,
                                          self.covariate_matrix,
                                          self.use,
                                          self.group_scaling)
        execute_br.performBayesianAnalysis()
        self.posterior_probs = np.array(execute_br.getPosteriorProbabilities())
        self.class_probs = np.array(execute_br.getPiEstimates())
        self.all_lnbfs = np.array(execute_br.getAllBFs())

    def posterior_analysis():
        pass

# TODO: Design a function to process all variables for Bayesian regression.
