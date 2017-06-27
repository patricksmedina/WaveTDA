import matplotlib.pyplot as plt
import numpy as np

class PersistenceDiagram(object):
    """Tools for managing persistence diagrams"""

    def __init__(self, PD):
        # TODO: Check that the persistence diagram is of the form we need.
        #   1) Dimensions = 3
        #   2) Numpy type
        #   3) Numeric type
        self.PD = PD
        self.homology_groups = np.unique(self.PD[:,0])
        self.max_homology_values = []

        for i in self.homology_groups:
            idx = np.where(self.PD[:,0] == i)[0]
            self.max_homology_values.append(np.max(self.PD[idx, 2]))

    def select_homology(self, homology_group, delete_rows=None):
        """
        Returns a new numpy array with the selected homology group.

        Parameters
        ----------
        homology_group : int
          Indicate which sub-arrays to remove.
        delete_rows : int, list or tuple, optional
          The index or indices of the reduced array to be deleted.

        Returns
        -------
        out : ndarray
            A copy of `PD` with rows not equal to `homology_group` removed.
            Additional elements of the reduced array are removed by `skiprows.`

        Examples
        --------
        >>> from wavetda import persistence_diagram as pd
        >>> myPD = np.array([[0,0,1], [0,0,0.5], [1,0.5,0.75]])
        >>> PD = pd.PersistenceDiagram(PD=myPD)
        >>> PD.select_homology(homology_group=0)
        array([[ 0. , 0. , 1.0 ],
               [ 0. , 0. , 0.5]])
        >>> PD.select_homology(homology_group=0,delete_rows=0)
        array([[ 0. , 0. , 0.5 ]])
        >>> PD.select_homology(homology_group=0,delete_rows=[0,1])
        array([], shape=(0, 3), dtype=float64)
        """

        tempPD = self.PD[np.where(self.PD[:,0] == homology_group)[0], :]

        if delete_rows is not None:
            tempPD = np.delete(tempPD, delete_rows, 0)

        return(tempPD)

    def plot(self, show_homgrp=None):
        """
        Displays a plot of the persistence diagram.

        Parameters
        ----------
        show_homgrp : list
          List containing the homology groups to plot.
        """

        plt.style.use('ggplot')

        if show_homgrp is None:
            show_homgrp = list(self.homology_groups)

        show_homgrp.sort()
        hommarkers = ["o", "^", "s"]
        homcolors = ["black", "red", "blue"]

        # cycle through homology groups and plot.
        fig, ax = plt.subplots()
        for hom in show_homgrp:
            col = homcolors[int(hom)]
            mark = hommarkers[int(hom)]
            tempPD = self.select_homology(homology_group=hom)[:,1:]
            ax.plot(tempPD[:,0],
                    tempPD[:,1],
                    alpha=0.75,
                       marker=mark,
                       color=col,
                       label="Hom {}".format(int(hom)),
                       ls="none")

        # plot features custom to the persistence diagram
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')

        # plot the diagonal line
        xmin = min(0, np.min(self.PD[:,1]))
        xmax = max(self.max_homology_values)
        ax.plot([xmin,xmax], [xmin,xmax], 'k--')

        # plot the legend
        ax.legend(loc=4, facecolor="white")

        # show the plot
        plt.show()
