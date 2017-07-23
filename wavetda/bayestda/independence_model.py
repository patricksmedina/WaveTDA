# import local modules
from wavetda.bayestda import bayes_regression_ind as br

# import package dependencies
import numpy as np
import os

class BayesianRegression(object):
    """
    BayesTDA independence model.

    Parameters
    ----------
    num_kernels : int
        The number of persistence kernels in the analysis.
    num_samples : int
        The number of samples.
    num_wavelets : int
        The total number of wavelet coefficients per sample.
    num_covariates : int
        The number of covariates for this analysis.
    num_scales : int
        The number of scales used in for the Haar wavelet basis.
    use : ndarray, optional, shape (num_wavelets, ) or (1, num_wavelets)
        Binary array indicating which wavelet coefficients to perform
        the analysis on.  Binary array elements with value zero will skip the
        analysis on those wavelet coefficients.  If value is none, the array
        will be computed from the data provided.

    Attributes
    --------

    """
    def __init__(
            self,
            num_kernels,
            num_samples,
            num_wavelets,
            num_covariates,
            num_scales,
            use=None
        ):

        self.num_kernels = num_kernels
        self.num_samples = num_samples
        self.num_wavelets = num_wavelets
        self.num_scales = num_scales
        self.num_covariates = num_covariates
        self.use = use

    def variable_error_checking():
        pass

    def fit(self, W, X):
        """
        Fits the linear model W = Xb + e using the Bayesian regression
        method outlined in [...].  Stores the results as attribute matrices.

        Parameters
        ----------
        W : ndarray, shape (num_samples, num_wavelets)
          Matrix of wavelet coefficients.
        X : ndarray, shape (num_covariates, num_samples)
          Matrix of covariates.


        Attributes
        -------
        out : ndarray
            A copy of `PD` with rows not equal to `homology_group` removed.
            Additional elements of the reduced array are removed by `skiprows.`

        Examples
        --------
        """

        if self.use is None:
            print(
                "[INFO] No use matrix specified.
                 Constructing default use matrix from the wavelet
                 coefficients."
            )

            # use only non-zero wavelet coefficients
            self.use = np.array(
                (np.sum(wavelet_coefficients, axis=1) > 0) * 1
            )
            self.use = self.use.reshape(1, -1)

        variable_error_checking()

        # create bayesRegression object
        execute_br = br.bayesRegression(
            self.num_scales,
            self.num_wavelets,
            self.num_covariates,
            self.num_samples,
            self.W,
            self.X,
            self.use
        )

        # fit the model
        execute_br()

        #
        self.posterior_probs = np.array(execute_br.getPosteriorProbabilities())
        self.class_probs = np.array(execute_br.getPiEstimates())
        self.all_lnbfs = np.array(execute_br.getAllBFs())
