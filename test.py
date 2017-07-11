import matplotlib.pyplot as plt

from wavetda.statistics import bayes_regression as br
import numpy as np
import shutil
import os

num_samples = 1000
num_kernels = 3
num_scales = 1
num_wavelets = 4 ** num_scales
num_covariates = 1
path = "/Users/patrickmedina/Desktop/test"

test_wav = np.zeros((num_kernels, num_samples, num_wavelets)).astype(np.float64)
test_cov = np.random.binomial(n=1, p=0.5, size=(num_samples, num_covariates)).astype(np.float32)

beta = 0.0
for row in range(num_samples):
    test_wav[:, row, :] = np.random.normal(loc=(beta * test_cov[row, 0]),
                                           scale=1.0,
                                           size=(num_kernels, num_wavelets))

use = np.ones(num_wavelets, dtype=int)


# construct the output directory
if os.path.exists(path):
    shutil.rmtree(path)

os.makedirs(path)
os.makedirs(os.path.join(path,"lnBFs"))
os.makedirs(os.path.join(path,"piEstimates"))
os.makedirs(os.path.join(path,"parameters"))

brtest = br.bayes_regression(num_samples,
                             num_kernels,
                             num_scales,
                             num_wavelets,
                             num_covariates,
                             wavelet_coeffs=test_wav.astype(np.float64),
                             covariates=test_cov,
                             use=use,
                             outdir=path,
                             ss_a = 0.4)

brtest()
brtest.getLnLikelihoodRatio()
