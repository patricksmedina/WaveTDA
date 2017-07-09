from wavetda.statistics import bayes_regression as br
import numpy as np
import shutil
import os

num_samples = 10
num_kernels = 5
num_scales = 5
num_wavelets = 4 ** num_scales
num_covariates = 10
path = "/Users/patrickmedina/Desktop/test"

test_wav = np.random.randint(0,256,(num_kernels,num_samples,num_wavelets)).astype(np.float32)
test_cov = np.random.standard_normal((num_samples,num_covariates)).astype(np.float32)
use = np.ones(num_wavelets, dtype=int)

# construct the output directory
if os.path.exists(path):
    shutil.rmtree(path)

os.makedirs(path)
os.makedirs(os.path.join(path,"lnBFs"))

brtest = br.bayes_regression(num_samples,
                             num_kernels,
                             num_scales,
                             num_wavelets,
                             num_covariates,
                             wavelet_coeffs=test_wav,
                             covariates=test_cov,
                             use=use,
                             outdir=path)

brtest() #.expectationMaximization()
