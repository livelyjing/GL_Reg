import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import GL_Reg as glr

print("All Packages seem to be installed.")
print("Running small test. This may take a minute.")

G_BA = nx.barabasi_albert_graph(5,1,seed = 777)
L_BA = nx.laplacian_matrix(G_BA).toarray()
normL_BA = (L_BA.shape[0]/np.trace(L_BA))*L_BA

Signals, Regressors = glr.genNormal(G_BA)

Results = glr.findHParams(Signals, Regressors, normL_BA, steps = 1)


