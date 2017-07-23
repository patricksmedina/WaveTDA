import numpy as np
import os

### GLOBAL VARIABLES
FILEDIR = os.path.abspath(os.path.dirname(__file__))

### DATA LOADING FUNCTIONS
def linked_annuli():
    """
    Loads the linked annuli data for hypothesis testing.
    """

    annuli = []
    files = os.listdir(os.path.join(FILEDIR,"linked_annulus"))

    for f in files:
        if(f.endswith(".npy")):
            annuli.append(
                np.load(
                    os.path.join(FILEDIR,"linked_annulus",f)
                )
            )
    return(annuli)

def annuli_regression():
    """
    Loads the annuli regression data in a dictionary.  Dictionary keys
    correspond to the value of the variables used in the regression.

    Character 1: 1 if Annulus, 0 if Disk.
    Character 2: 1 if objects separated, 0 if connected.
    Character 3: 1 if one object is small, 0 if both are large.

    For example: two unlinked annuli with two large objects has key '110'.
    """

    annuli = []
    ann_number = []
    files = os.listdir(os.path.join(FILEDIR,"annuli_regression"))

    for f in files:
        if(f.endswith(".npy")):
            annuli.append(
                np.load(
                    os.path.join(FILEDIR,"annuli_regression",f)
                )
            )

            f = f.split("_")[-1]
            f = f.split(".")[0]
            ann_number.append(f)

    temp_ann = zip(ann_number, annuli)
    temp_ann.sort()
    return(dict(temp_ann))
