import numpy as np
import os

FILEDIR = os.path.abspath(os.path.dirname(__file__))

def load_linked_annulus():
    """

    """

    annuli = []
    files = os.listdir(os.path.join(FILEDIR,"linked_annulus"))

    for f in files:
        if(f.endswith(".npy")):
            annuli.append(np.load(os.path.join(FILEDIR,"linked_annulus",f)))

    return(annuli)
