# import scipy.stats as st
import numpy as np
import pywt

class PersistenceWavelets(object):

    def __init__(self, scale):
        self.scale = scale
        self.basis = "haar"

    def construct_wavelet_vector(self, data):
        """
        construct2DWaveletVector
        """

        # Compute the 2D Haar wavelet coefficients for all scales
        # Tuple order across locations is ([Vertical], [Horizontal], [Diagonal])
        coeff = pywt.wavedec2(data, self.basis)

        # Merge the coefficients into a single array, while preserving the order of the input
        coeff_vector = np.hstack([np.vstack(coeff[i]).flatten() for i in range(0,len(coeff))])
        coeff_level_size = np.vstack([(np.vstack(coeff[i]).flatten()).shape[0] for i in range(0,len(coeff))])

        return(coeff_vector)

    def reconstruct_wavelet_array(coeff_vector):
        """
        Reconstructs the PyWavelets vector.
        """

        # assign an empty wavelet coefficient list
        wavelet_coeffs = []

        #
        ### Store the wavelet coefficient at the largest scale
        #
        # assign the largest coefficient vector
        wavelet_coeffs.append(np.array(coeff_vector[0], ndmin = 2))
        #
        # remove the coefficient
        coeff_vector = np.delete(coeff_vector, 0)
        #
        ### Store the wavelet coefficients at smaller scales
        s = 0
        while coeff_vector.shape[0] > 0:
            scale_tuple = ()
            dim = 2 ** s
            # build the coefficient array at scale (s + 1)
            for i in np.arange(1,4):
                scale_tuple += (coeff_vector[(i-1)*(dim ** 2):i*(dim ** 2)].reshape(dim, dim), )

            # add tuple to the wavelet array
            wavelet_coeffs.append(scale_tuple)

            # delete the coefficients
            coeff_vector = np.delete(coeff_vector, np.arange(0,3*(dim**2)))

            # iterate scale
            s += 1

        return(wavelet_coeffs)




# def simulateWaveletCoefficients(mean, var, phi, dof, samples, seed = None):
#     """
#     simulateWaveletCoefficients
#     """
#     # set seed
#     np.random.seed(seed)
#
#     # Return zero samples if variance is zero.
#     if var is 0:
#         return np.repeat(0.0, samples)
#
#     # mixture component
#     t = np.random.binomial(n = 1, p = phi, size = samples)
#
#     # non-standardized t rvs
#     x = st.t.rvs(df = dof, loc = mean, scale = np.sqrt(var), size = samples)
#
#     # zero out values from degenerate distribution
#     x *= t
#
#     # return
#     return(x)


# def convertWaveletsToData(wcs, basis, **kwargs):
#     """
#     convertWaveletsToData
#     """
#     return(pywt.waverec2(wcs, basis, **kwargs))
# # END FUNCTION
#
# if __name__ == "__main__":
#     # random data
#     data = np.random.normal(loc=0, scale=1.0, size = 1024).reshape(32,32)
#
#     # wavelet coefficients
#     mywc = construct2DWaveletVector(data = data,
#                                     basis = "haar",
#                                     returnCoefficientIDs = False)
#
#     # wavelet coefficients and the waveqtl group ids
#     mywc2, grp_ids = construct2DWaveletVector(data = data,
#                                               basis = "haar",
#                                               returnCoefficientIDs = True)
#
#
#     print mywc
#     print mywc2
#     print all(mywc == mywc2)
#     print grp_ids
