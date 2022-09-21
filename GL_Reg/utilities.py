import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io 

import GL_Reg.signal_gen as glr

# Quickly plot a laplacian
def plotLap(lap):
    adj = abs(lap) - np.diag(np.diag(lap))
    G = nx.from_numpy_matrix(adj)
    nx.draw_circular(G, node_size=1000, with_labels = True, 
                     edgecolors='blue', node_color='lightgray')

def saveData(file_name, signal):
    D = {"S": signal}
    scipy.io.savemat(file_name, D)
    
def genNormal(G, beta = [1, 1], num_signals = 100, 
              mean = 0, sd = 10, mu = 0, sigma = .5, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_norm = np.random.normal(mean, sd, size= size)
    P_norm = [ONES, X_norm]
    S_norm = glr.RandomRegressorSignal2(G, size[1], mu, sigma, beta, P_norm, seed)[1]
    
    return((S_norm, P_norm))

def genBinomial(G, beta = [1, 1], num_signals = 100, p = .5, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_binom = np.random.binomial(1, 0.5, size=size)
    P_binom = [ONES, X_binom]
    S_binom = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_binom, seed)[1]
    
    return((S_binom, P_binom))
    

def genExp(G, beta = [1, 1], num_signals = 100, scale = 1, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_exp = np.random.exponential(scale, size = size)
    P_exp = [ONES, X_exp]
    S_exp = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_exp, seed)[1]
    
    return((S_exp, P_exp))

def genPoisson(G, beta = [1, 1], num_signals = 100, lam = 1, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_pois = np.random.poisson(lam, size = size)
    P_pois = [ONES, X_pois]
    S_pois = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_pois, seed)[1]
    
    return((S_pois, P_pois))

def genGamma(G, beta = [1, 1], num_signals = 100, shape = 1, scale = 1, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_gamma = np.random.gamma(shape, scale, size = size)
    P_gamma = [ONES, X_gamma]
    S_gamma = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_gamma, seed)[1]
    
    return((S_gamma, P_gamma)) 

def genPower(G, beta = [1, 1], num_signals = 100, a = 5, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_pow = np.random.power(a, size = size)
    P_pow = [ONES, X_pow]
    S_pow = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_pow, seed)[1]
    
    return((S_pow, P_pow))

def genRayleigh(G, beta = [1, 1], num_signals = 100, scale = 1, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_ray = np.random.rayleigh(scale, size = size)
    P_ray = [ONES, X_ray]
    S_ray = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_ray, seed)[1]
    
    return((S_ray, P_ray))


def genAllDist(G, beta = [1, 1], num_signals = 100, seed = 123):
    np.random.seed(seed = seed)
    size = (G.number_of_nodes(), num_signals)
    ONES = np.ones(size)
    
    X_norm = np.random.normal(0, 10, size= size)
    P_norm = [ONES, X_norm]
    S_norm = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_norm, 123)[1]

    X_binom = np.random.binomial(1, 0.5, size=size)
    P_binom = [ONES, X_binom]
    S_binom = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_binom, 123)[1]

    X_exp = np.random.exponential(1, size = size)
    P_exp = [ONES, X_exp]
    S_exp = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_exp, 123)[1]

    X_pois = np.random.poisson(1, size = size)
    P_pois = [ONES, X_pois]
    S_pois = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_pois, 123)[1]

    X_gamma = np.random.gamma(1,1, size = size)
    P_gamma = [ONES, X_gamma]
    S_gamma = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_gamma, 123)[1]

    X_pow = np.random.power(5, size = size)
    P_pow = [ONES, X_pow]
    S_pow = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_pow, 123)[1]

    X_ray = np.random.rayleigh(1, size = size)
    P_ray = [ONES, X_ray]
    S_ray = glr.RandomRegressorSignal2(G, size[1], 0, 0.5, beta, P_ray, 123)[1]

    Signals = [S_norm, S_binom, S_exp, S_pois, S_gamma, S_pow, S_ray]
    Regressors = [P_norm, P_binom, P_exp, P_pois, P_gamma, P_pow, P_ray]
    return((Signals, Regressors))
    
    
    