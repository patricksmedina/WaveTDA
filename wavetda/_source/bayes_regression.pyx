###
###
###

# external packages
cimport cnumpy
import numpy

# declare fused types for wavelet and covariate arrays
ctypedef fused wavetype:
    float
    long

ctypedef fused covatype:
    int
    float
    long



# cython class for bayes regression independence model
cdef class BayesRegressionIM:
    def __cinit__(self,
                  int num_samples,
                  int num_kernels,
                  int num_wavelets,
                  int num_covariates):

        self.bayesReg = new bayesRegressionDM(numScales,
                                              numWavelets,
                                              numCovariates,
                                              numSamples,
                                              waveletCoefficients,
                                              covariateMatrix,
                                              use,
                                              groupScaling)

    def __dealloc__(self):
        del self.bayesReg

    def performBayesianAnalysis(self):
        self.bayesReg.performBayesianAnalysis()

    def getPosteriorProbabilities(self):
        return self.bayesReg.getPosteriorProbabilities()

    def getPiEstimates(self):
        return self.bayesReg.getPiEstimates()

    # TODO: DELETE WHEN DONE!!!
    def getAllBFs(self):
        return self.bayesReg.getAllBFs()
