# disutils: language = c++

# Cython declaration of bayesRegression classes in bayestda.h

# import cython support for C++ vector in STL
from libcpp.vector cimport vector
from libcpp cimport bool

# Declaring C++ class interface to Cython
cdef extern from "_bayesregression.h" namespace "btda":

    # Declaring C++ class for the dependence model
    cdef cppclass bayesRegressionDM:
        # Public
        # declare constructor
        bayesRegressionDM(int, int, int, int, vector[vector[float]] &, vector[vector[float]] &, vector[int] &, bool) except +

        # perform bayesian analysis
        void performBayesianAnalysis()

        # access routines
        vector[vector[double]] getPosteriorProbabilities()
        vector[double] getPiEstimates()

        # TODO: DELETE WHEN DONE!!!
        vector[vector[double]] getAllBFs()

# Cython wrapper for C++ Class
cdef class BayesRegressionDM:
    cdef bayesRegressionDM *bayesReg

    def __cinit__(self,
                  int numScales,
                  int numWavelets,
                  int numCovariates,
                  int numSamples,
                  vector[vector[float]] & waveletCoefficients,
                  vector[vector[float]] & covariateMatrix,
                  vector[int] & use,
                  bool groupScaling = False):

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
