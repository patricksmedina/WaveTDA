from wavetda.statistics import bayestda as bayestda

import matplotlib.pyplot as plt
import numpy as np

def matthews_simulations(numSamples, numWavelets, b0, b1, r):
    wc0, wc1, x = [], [], []
#
    for i in range(numSamples):
        x.append(np.random.binomial(2, 0.5, 1)[0])
#
        for _ in range(numWavelets):
#
            w = np.random.multivariate_normal(mean=(b0 * x[i], b1 * x[i]),
                                              cov=[[0.5,r],[r,0.5]],
                                              size=1)
            wc0.append(w[0][0])
            wc1.append(w[0][1])
#
    wc0 = np.array(wc0).reshape(-1, numWavelets)
    wc1 = np.array(wc1).reshape(-1, numWavelets)
    return(np.hstack((wc0, wc1)), np.array(x).reshape(-1, 1))

numSamples = 1000
numWavelets = 4 ** 3
numCovariates = 1
groupScaling = False
numScales = np.log(int(numWavelets) / 4)

b0, b1 = 0, -1
r = 0.4

waveletCoefficients, covariateMatrix = matthews_simulations(numSamples,
                                                            numWavelets,
                                                            b0, b1, r)
use = np.array([1] * (2 * numWavelets))

br = bayestda.BayesianRegression(waveletCoefficients, covariateMatrix, numSamples, numWavelets, numCovariates, numScales, use, groupScaling)
br()
# # #
print(np.array(br.class_probs).reshape(-1,4))
# allLnBayesFactors = br.all_lnbfs

# plt.figure(1)
# idx2 = np.where(np.tile(covariateMatrix,numWavelets).ravel() == 2)[0]
# idx1 = np.where(np.tile(covariateMatrix,numWavelets).ravel() == 1)[0]
# idx0 = np.where(np.tile(covariateMatrix,numWavelets).ravel() == 0)[0]
#
# wc0 = waveletCoefficients[:, :numWavelets]
# wc1 = waveletCoefficients[:, numWavelets:]
# plt.subplot(221)
# plt.scatter(wc1.ravel()[idx2], wc0.ravel()[idx2],c = "green")
# plt.scatter(wc1.ravel()[idx1], wc0.ravel()[idx1],c = "orange")
# plt.scatter(wc1.ravel()[idx0], wc0.ravel()[idx0],c = "blue")
# plt.subplot(222)
# plt.scatter(np.tile(covariateMatrix,numWavelets).ravel(), wc0.ravel())
# plt.subplot(223)
# plt.scatter(np.tile(covariateMatrix,numWavelets).ravel(), wc1.ravel())
# plt.show()
#
# n_bins = 20
# plt.figure(2)
# plt.subplot(131)
# plt.hist(allLnBayesFactors[0, :], bins=n_bins)
# plt.subplot(132)
# plt.hist(allLnBayesFactors[1, :], bins=n_bins)
# plt.subplot(133)
# plt.hist(allLnBayesFactors[2, :], bins=n_bins)
# plt.show()
