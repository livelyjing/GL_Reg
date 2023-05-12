import numpy as np
from sklearn import metrics

# get the matrix dimension and lower triangular part
def lowertrian(Lap):
    N = Lap.shape[0]
    lowertri = Lap[np.tril_indices(N, k = -1)]
    return(lowertri)

# The first argument represents the ground truth Laplacian, the second one represents the learned Laplacian

def metricsprf(L0, L):
    #L_0tmp = L_0-diag(diag(L_0));
    
    L_0tmp = lowertrian(L0)
    
    #edges_groundtruth = squareform(L_0tmp)~=0;
    edges_groundtruth = L_0tmp != 0;
    
    #Ltmp = L-diag(diag(L));
    #edges_learned = squareform(Ltmp)~=0;
    
    L_tmp = lowertrian(L)
    
    edges_learned = L_tmp != 0;
    
    
    if sum(edges_learned) != 0:
        precision = metrics.precision_score(edges_groundtruth, edges_learned)
        recall = metrics.recall_score(edges_groundtruth, edges_learned)
        fscore = metrics.f1_score(edges_groundtruth, edges_learned)
        nmi = metrics.normalized_mutual_info_score(edges_groundtruth, edges_learned)
    else:
        precision = -1
        recall = -1
        fscore = -1
        nmi = -1
    return(precision, recall, fscore, nmi)
    
