import scipy.stats as st
import numpy as np

def _simulate_wavelet_coefficients(mean, var, phi, dof, samples = 1000, seed = None):
    """
    simulateWaveletCoefficients
    """

    # generate a seed
    if seed is None
        seed = np.random.randint(0, 2 ** 32 - 1, 1)[0]

    np.random.seed(seed)

    # Return zero samples if variance is zero.
    if var is 0:
        return np.repeat(0.0, samples)

    # mixture component
    t = np.random.binomial(n = 1, p = phi, size = samples)

    # non-standardized t rvs
    x = st.t.rvs(df = dof, loc = mean, scale = np.sqrt(var), size = samples)

    # zero out values from degenerate distribution
    x *= t

    # return
    return(x)

def model_effect_size(pk, cv, num_samples, outdir, ):
    """
    Models the effect size for a given
    """
    pass
