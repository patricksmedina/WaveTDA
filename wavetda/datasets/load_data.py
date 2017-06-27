# import wavetda.datasets
# import numpy as np
# import os
#
# def load_linked_annulus():
#     """
#
#     """
#     annuli = []
#     files = os.listdir(os.path.join(os.path.dirname(__file__),"linked_annulus"))
#     for f in files:
#         if(f.endswith(".npy")):
#             dirname = os.path.abspath(__file__).split("/")[:-1]
#             annuli.append(np.load(os.path.join(dirname + [f])))
#
#     return(annuli)
