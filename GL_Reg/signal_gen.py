import networkx as nx
import numpy as np
from scipy import sparse
from numpy.linalg import inv
import itertools as it
from numpy.linalg import eig

def RandomRegressorSignal2(graph, num_signals, mu, sigma, b, P, myseed):
    # The Graph Laplacian
    L = nx.laplacian_matrix(graph).toarray()
    
    # The number of vertices
    size = L.shape[0]
    NormL = (size/np.trace(L))*L
    D,V= np.linalg.eig(NormL)
    covh = np.linalg.pinv(np.diag(D), hermitian= True) 
    my_mean = mu*np.ones(size)
    # R = b[0]*P[0]+b[1]*P[1]+b[2]*P[2]+b[3]*P[3]
    R = sum([P[i]*b[i] for i in range(len(P))])

    # R = b[0]*P[0]+b[1]*P[1]
    np.random.seed(seed = myseed)

    gftcoeff = np.random.multivariate_normal(my_mean, covh, num_signals)
    X = V@gftcoeff.T + R
    
    np.random.seed(seed = myseed+1)

    X_noisy = X + sigma*np.random.normal(0, 1, size=X.shape)
    return [X, X_noisy]