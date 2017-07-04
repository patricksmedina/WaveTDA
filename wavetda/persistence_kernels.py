# external packages
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
import numpy as np

class PersistenceKernel(object):
    """ Routines for computing persistence kernels """

    def __init__(self, PD, birth_domain, death_domain, scale):
        self.PD = PD
        self.birth_domain = birth_domain
        self.death_domain = death_domain
        self.scale = scale

    def _construct_domain(self):
        bd = np.linspace(start = self.birth_domain[0],
                         stop = self.birth_domain[1],
                         num = 2 ** self.scale)

        dd = np.linspace(start = self.death_domain[0],
                         stop = self.death_domain[1],
                         num = 2 ** self.scale)
        return(np.meshgrid(bd, dd))

class PWGK(PersistenceKernel):

    def __init__(self, PD, birth_domain, death_domain, scale, C, p, bandwidth):
        super(PWGK, self).__init__(PD, birth_domain, death_domain, scale)
        self.C = C
        self.p = p
        self.bandwidth = bandwidth

    def smooth(self, homology_group=None, delete_rows=None, zero_out=False):
        # construct the domain
        xx, yy = self._construct_domain()
        domain = np.vstack((xx.ravel(), yy.ravel())).T

        # copy PD from the PD object -- isolate homology group
        if homology_group is None:
            temp_PD = self.PD.PD.copy()
        else:
            temp_PD = self.PD.select_homology(homology_group=homology_group,
                                              delete_rows=delete_rows)

        # Return zero matrix if homology group is empty
        if temp_PD.shape[0] == 0:
            return(np.zeros((2 ** self.scale,
                             2 ** self.scale)))

        # Isolate the persistence diagram to just the relevant birth-death points.
        if temp_PD.shape[1] == 3:
            temp_PD = temp_PD[:, 1:]

        # pairwise subtraction between grid points and each point in the persistence diagram
        # reshape from tensor to matrix (K x 2)
        Fbd = np.subtract(domain[:, np.newaxis, :],
                          temp_PD[np.newaxis, :, :]).reshape(-1, 2)
        Fbd = np.square(np.linalg.norm(Fbd, axis = 1)).reshape(domain.shape[0],-1)

        # for each persistence point, compute the Gaussian at each point in the domain
        Fbd = np.exp(- Fbd / (2.0 * self.bandwidth ** 2))
        Fbd /= (2 * np.pi * self.bandwidth ** 2)

        # compute the weights for each birth-death point
        weights = np.arctan(self.C * (temp_PD[:, 1] - temp_PD[:, 0]) ** self.p)

        Fbd = np.sum(np.multiply(weights, Fbd), axis = 1)

        # zero out stuff below the diagonal
        if zero_out:
            Fbd[np.where(domain[:,0] >= domain[:,1])] = 0.0

        # return the kernel between PD1 and PD2
        return(Fbd.reshape(2 ** self.scale, 2 ** self.scale))

class PIF(PersistenceKernel):
    def __init__(PD, birth_domain, death_domain, scale, bandwidth):
        self.bandwidth = bandwidth
        super(PIF, self).__init__(PD, birth_domain, death_domain, scale)

    def smooth(self, homology_group=None, delete_rows=None, zero_out=False):

        # construct the domain
        xx, yy = self._construct_domain()
        domain = np.vstack((xx.ravel(), yy.ravel())).T

        # copy PD from the PD object -- isolate homology group
        if homology_group is None:
            temp_PD = self.PD.PD.copy()
        else:
            temp_PD = self.PD.select_homology(homology_group=homology_group,
                                             delete_rows=delete_rows)

        # Return zero matrix if homology group is empty
        if temp_PD.shape[0] == 0:
            return(np.zeros((2 ** self.scale,
                             2 ** self.scale)))

        # Isolate the persistence diagram to just the relevant birth-death points.
        if temp_PD.shape[1] == 3:
            temp_PD = temp_PD[:, 1:]

        # pairwise subtraction between grid points and each point in the persistence diagram
        # reshape from tensor to matrix (K x 2)

        Fbd = np.subtract(domain[:, np.newaxis, :], temp_PD[np.newaxis, :, :]).reshape(-1, 2)
        Fbd = np.square(np.linalg.norm(Fbd, axis = 1)).reshape(domain.shape[0],-1)

        # compute weights
        weight = temp_PD[:, 1] - temp_PD[:, 0]

        # for each persistence point, compute the Gaussian at each point in the domain
        Fbd = np.exp(- Fbd / (2.0 * self.bandwidth ** 2)) \
              / (2 * np.pi * self.bandwidth ** 2)
        Fbd = np.sum(np.multiply(weight, Fbd), axis = 1)

        # zero out stuff below the diagonal
        if zero_out and not PD.rotated:
            Fbd[np.where(domain[:,0] >= domain[:,1])] = 0.0

        # store the kernel smoothed diagram
        return(Fbd.reshape(2 ** self.scale, 2 ** self.scale))

class PI(PersistenceKernel):
    def __init__(PD, birth_domain, death_domain, scale, bandwidth,
                 max_persistence=None):
        self.bandwidth = bandwidth
        self.max_persistence = max_persistence
        super(PI, self).__init__(PD, birth_domain, death_domain, scale)

    def smooth(self, homology_group=None, delete_rows=None):

        # construct domain
        bd = np.linspace(start = self.birth_domain[0],
                         stop = self.birth_domain[1],
                         num = 2 ** self.scale + 1)
        pd = np.linspace(start = self.death_domain[0],
                         stop = self.death_domain[1],
                         num = 2 ** self.scale + 1)

        # copy PD from the PD object -- isolate homology group
        if homology_group is None:
            temp_PD = self.PD.PD.copy()
        else:
            temp_PD = self.PD.select_homology(homology_group=homology_group,
                                              delete_rows=delete_rows)

        # rotate the persistence diagram and reduce to relevant info.
        temp_PD[:,2] = temp_PD[:,2] - temp_PD[:,1]
        temp_PD = temp_PD[:, 1:]

        # Return zero matrix if homology group is empty
        if temp_PD.shape[0] == 0:
            return(np.zeros((2 ** self.scale,
                             2 ** self.scale)))

        # Row i: [(domain_i - b_0) / bandwidth,..., (domain_i - b_N) / bandwidth]
        normalized_birth = np.subtract(bd[:, np.newaxis],
                                       temp_PD[:, 0].reshape(1,-1))
        normalized_birth = normalized_birth / float(bandwidth)

        # Row i: [(domain_i - p_0) / bandwidth,..., (domain_i - p_N) / bandwidth]
        normalized_persistence = np.subtract(pd[:, np.newaxis],
                                             temp_PD[:, 1].reshape(1,-1))
        normalized_persistence = normalized_persistence / float(bandwidth)

        pointwise_birth_probs = norm.cdf(normalized_birth, 0, 1)
        pointwise_persistence_probs = norm.cdf(normalized_persistence, 0, 1)

        diff_birth_probs = np.diff(pointwise_birth_probs, axis = 0)
        diff_persistence_probs = np.diff(pointwise_persistence_probs, axis = 0)

        # compute weights for each (birth, persistence) point
        if max_persistence is None:
            weights = temp_PD[:,1] / np.max(temp_PD[:,1])
        else:
            weights = temp_PD[:,1] / float(max_persistence)

        # apply weights to persistence differences
        diff_persistence_probs = np.multiply(weights, diff_persistence_probs)

        # allocate memory for pixel array
        ncols = 2 ** self.scale
        pixels = np.zeros((ncols, ncols), dtype = np.float64)

        # compute each pixel
        for i in range(ncols):
            for j in range(ncols):
                pixels[j, i] = np.multiply(diff_birth_probs[i, :],
                                           diff_persistence_probs[j, :]).sum()

        return(np.flipud(pixels))

#
# def plot_kernel(self, f, show_contours=False, cmap="Blues", n_levels=11, PD=False, homology_group=None):
#
#     def plot_pi(self, f, cmap, PD, homology_group):
#         f = np.flipud(f)
#         fig, ax = plt.subplots()
#
#         # set the domain
#         extent = [self.birth_domain[0], self.birth_domain[1], self.death_domain[0], self.death_domain[1]]
#         cset = ax.imshow(f, interpolation="nearest", cmap=cmap, origin="lower",
#                          extent=extent)
#
#         # overlay the persistence diagram
#         if PD:
#             if homology_group is None:
#                 homology_group = list(self.PD.homology_groups)
#
#             homology_group.sort()
#             hommarkers = ["o", "^", "s"]
#             homcolors = ["black", "red", "blue"]
#
#             # cycle through homology groups and plot.
#             for hom in homology_group:
#                 col = homcolors[int(hom)]
#                 mark = hommarkers[int(hom)]
#                 tempPD = self.PD.select_homology(homology_group=hom)
#                 tempPD[:, 2] = tempPD[:, 2] - tempPD[:, 1]
#                 tempPD = tempPD[:, 1:]
#
#                 ax.plot(tempPD[:,0],
#                         tempPD[:,1],
#                         marker=mark,
#                         color=col,
#                         label="Hom {}".format(int(hom)),
#                         ls="none")
#             ax.legend()
#
#         ax.set_xlabel('Birth')
#         ax.set_ylabel('Persistence')
#
#         plt.colorbar(cset)
#
#         plt.show()
#
#
#     def plot_fn(self, f, show_contours, cmap, n_levels, PD, homology_group):
#         x, y = self._construct_domain()
#
#         # plt.style.use('ggplot')
#
#         l = np.linspace(0, np.max(f), n_levels)
#
#         fig, ax = plt.subplots()
#         cset = ax.contourf(x, y, f, cmap = cmap, levels=l)
#
#         ax.set_xlabel('Birth')
#         ax.set_ylabel('Death')
#
#         # plot the diagonal line
#         xmin = min(0, np.min(self.PD.PD[:,1]))
#         xmax = max(np.max(x), np.max(y))
#         ax.plot([xmin,xmax], [xmin,xmax], '--', c="0.80", lw=0.90)
#
#         # overlay the persistence diagram
#         if PD:
#             if homology_group is None:
#                 homology_group = list(self.PD.homology_groups)
#
#             homology_group.sort()
#             hommarkers = ["o", "^", "s"]
#             homcolors = ["black", "red", "blue"]
#
#             # cycle through homology groups and plot.
#             for hom in homology_group:
#                 col = homcolors[int(hom)]
#                 mark = hommarkers[int(hom)]
#                 tempPD = self.PD.select_homology(homology_group=hom)[:,1:]
#                 tempPD = tempPD.reshape(-1, 2)
#                 tempPD[:,1] = tempPD[:,1] - tempPD[:,0]
#                 ax.plot(tempPD[:,0],
#                         tempPD[:,1],
#                         marker=mark,
#                         color=col,
#                         label="Hom {}".format(int(hom)),
#                         ls="none")
#             ax.legend(loc=4)
#
#         # colorbar and show
#         plt.colorbar(cset)
#         plt.show()
#
#
#     if self.kernel == "pi":
#         plot_pi(self, f, cmap, PD, homology_group)
#
#     else:
#         plot_fn(self, f, show_contours, cmap, n_levels, PD, homology_group)
# levels = np.linspace(-2.0, 1.601, 40)
# norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
#
# fig, ax = plt.subplots()
# cset1 = ax.contourf(
#     X, Y, Z, levels,
#     norm=norm)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_xticks([])
# ax.set_yticks([])
#
#
#     """ Plot kernel functions with a filled contour plot """
#
#     class MidpointNormalize(Normalize):
#         def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#             self.midpoint = midpoint
#             Normalize.__init__(self, vmin, vmax, clip)
#
#         def __call__(self, value, clip=None):
#             # I'm ignoring masked values and all kinds of edge cases to make a
#             # simple example...
#             x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#             return np.ma.masked_array(np.interp(value, x, y))
#
#     # get the mesh grid.
#     xx, yy = self.construct_domain()
#
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.set_xlim(np.min(xx), np.max(xx))
#     ax.set_ylim(np.min(yy), np.max(yy))
#
#     if np.min(f) < 0 < np.max(f):
#         norm = MidpointNormalize(midpoint = 0)
#
#         if "kwargs" in locals():
#             kwargs.update({"vmax": np.max(f), "vmin":np.min(f), "norm":norm})
#         else:
#             kwargs = {"vmax": np.max(f), "vmin":np.min(f), "norm":norm}
#
#     if "kwargs" in locals():
#         kwargs.update({"vmax": np.max(f), "vmin":np.min(f)})
#     else:
#         kwargs = {"vmax": np.max(f), "vmin":np.min(f)}
#
#     cfset = ax.contourf(xx, yy, f, **kwargs)
#     plt.colorbar(cfset, orientation = 'vertical', shrink = 0.8)
#
#     # Contour plot
#     if show_contours:
#         if "kwargs" in locals():
#             if "cmap" in kwargs.keys():
#                 kwargs.pop("cmap")
#
#             kwargs.update({"colors": "k"})
#
#     cset = ax.contour(xx, yy, f, **kwargs)
#     ax.clabel(cset, inline = 1, fontsize = 10)
#
#     # Label plot
#     ax.set_xlabel('Birth')
#     ax.set_ylabel('Death')
#     plt.plot(np.array([np.min(xx), np.max(xx)]), np.array([np.min(yy), np.max(yy)]),'--k')
#
#     if PD is not None:
#         plt.scatter(PD[:, 0], PD[:, 1], **kwargs)
#
#     plt.show()
