import numpy as np
import cvxpy as cp

# Description of inputs

    # X_noisy: the observed signal (Num_NodesxNum_Siganls numpy array)
    # P      : the observed regressors/observed explanatory variable (List of Num_NodesxNum_Siganls numpy arrays for each regressor)
    #          the last numpy array is a constant array of 1's to account for y-intercept.
    # L      : The estimated laplacian (Num_NodesxNum_Nodes numpy array)
    # hatS   : The estimated signal (Num_NodesxNum_Signals numpy array)
    # b      : The estiamted regression coefficients (list of doubles)
    # w1     : Hyperparameter for controlling smoothness of the estimated signal over the learned graph (double)
    #          A HIGHER value forces estimated signal to be SMOOTHER over the estimated laplacian.
    # w2     : Hyperparameter for controlling how sparse the estimated laplacian is.
    #          A HIGHER value forces the estimated laplacian to be MORE sparse.

# Helper functions

# Make a numpy array into a row vector
# input: nxn array
# return: (n^2)x1 vector
# example: 
# < vec(numpy.array([[1,2],[3,4]])) 
# > matrix([[1],
#           [2],
#           [3],
#           [4]])
def vec(S):
    return(S.flatten('F').T)

# Multiplys the regression coefficent to the appropriate explanatory variable for every node. Every vector is added together and 
# then reshaped into a vector.

# input 1: P list of Num_NodesxNum_Signals (number of nodes in graph) represents the ith explanatory variable value on each node in the graph
# and each round of signals

# input 2: b is a list of regression coefficents. 
# output: (|G|*Num_Signals)x1 column vector

# example: 
# < P = [np.array([[1,2],[3,4]]),np.array([[5,6],[7,8]])]
# < b = [1,2]
# < Pbeta(P,b)
# > array([11, 17, 14, 20])

def Pbeta(P,b):
    R = sum([P[i]*b[i] for i in range(len(P))])
    vecR = vec(R)
    return(vecR)

# Helps reshape the list of regressor arrays into a matrix, so multiplication by regression coeffiecents is easy.

# input: List of Num_NodesxNum_Signals arrays for each explanatroy variable.
# output: (Num_Nodes*Num_Signal)x(Num_Regressors+1) array

# example:
# < P = [np.array([[1,2],[3,4]]),np.array([[5,6],[7,8]])] (2 nodes, 2 signals, 1 regressor 
# > array([[1, 5],
#          [3, 7],
#          [2, 6],
#          [4, 8]])

def pstack(P):
    Pvec = [p.ravel(order="F") for p in P]
    Pstack = np.stack(Pvec, axis = 1)
    return(Pstack)

# GL_Reg Algorithm

# Sets edge weights that are less than threshhold to 0.
def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

# Step 1: Fix S and beta, and update L.

# input 1: hatS is the previously learned signal estimation. Initialize to the 
#          observed signal array.
# input 2: P is the measured regressors (i.e. the observed explanatory variables)
# input 3: beta is the estimated regression coefficents
# input 4: w1 is the hyperparameter that controls smoothness penalty term
# input 5: w2 is the hyperparameter that penalizes sparsity of the learned Laplacian

# output: Updated estimate for Laplacian.

def optimize_L(hatS, P, beta, w1, w2):
    N = hatS.shape[0]
    n = hatS.shape[1]
    ONE = np.ones((N,1))
    ZERO = np.zeros((N,1))
    L = cp.Variable((N,N), symmetric = True)
    constraints = [
                   cp.trace(L) == N,                       
                   L - cp.diag(cp.diag(L)) <= 0,  # off diagnal elements 0, or negative, in which case they are equal. 
                   cp.diag(L) >= 0,
                   L@ONE == 0
                  ]
    x = vec(hatS) - Pbeta(P,beta)
    D = cp.kron(np.identity(n), L)
    objective = cp.Minimize(w1*cp.quad_form(x,D) + w2*(cp.norm(L, 'fro')**2))  ### Change to square 7-5-2022
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return(L.value)

# Step 2: Fix L and beta, and update hatS (learned Signal).

# input 1: L is the estimated Laplacian from the previous step
# input 2: P is the measured regressors (i.e. the observed explanatory variables)
# input 3: beta is the estimated regression coefficents
# input 4: S is the observed signal

# output: Updated estimate for estimated Signal hatS.

def optimize_hatS(L, P, beta, S, w1):
    N = S.shape[0]
    n = S.shape[1]
    InXL = np.kron(np.identity(n), L)
    term1 = w1*InXL + np.identity(N*n)
    term2 = np.linalg.inv(term1)
    term3 = vec(S) + w1*(InXL@Pbeta(P,beta))
    return((term2@term3).reshape((N,n), order = "F"))

# Step 3: Fix L and hatS and update estimated regression coefficients 
def optimize_beta(P, L, hatS):
    n = hatS.shape[1]
    InXL = np.kron(np.identity(n), L)
    Ptemp = pstack(P)
    term1 = Ptemp.T@InXL@Ptemp
    term2 = np.linalg.inv(term1)@Ptemp.T # inverse one will not work?
    term3 = InXL@vec(hatS)
    return(term2@term3)

# Full Graph Learning Regression algorithm

# input 1: X_noisy is the observed signal
# input 2: P is the observed regressors
# input 3: w1 is the hyperparameter that penalizes non smooth signals.
# input 4: w2 is the hyperparameter that penalizes non sparse laplacian. 
# input 4: the maximum iterations the algorithm will perform. 

# ouput: A list [L, hatS, beta] that has the estimated laplacian, signal, and regression coefficients. 

def GSPRegression(X_noisy, P, w1, w2, max_iter = 100, threshhold = 10**(-3)):
    N = X_noisy.shape[0]
    # We need a deep copy of X_noisy
    hatS_0 = X_noisy.copy()
    hatS = X_noisy.copy()
    n = hatS.shape[1]
    objective = [0]*max_iter
    beta = np.zeros((len(P),1))
    for i in range(max_iter):
        L = optimize_L(hatS,P,beta, w1, w2)
        hatS = optimize_hatS(L,P,beta,X_noisy,w1)
        beta = optimize_beta(P,L,hatS)
        # Construct the objective:
        arg1 = np.linalg.norm(hatS - hatS_0, 'fro')**2 
        #print("arg1", arg1)
        # Change the second part of objective function
        # arg2 = w1*(np.transpose((Y@np.transpose(Y)).flatten('F'))@(L.flatten('F')))
        x = vec(hatS) - Pbeta(P,beta)
        D = np.kron(np.identity(n), L)
        arg2 = w1*np.transpose(x)@D@x
        #print("arg2", arg2)
        arg3 = w2*np.linalg.norm(L, 'fro')**2
        #print("arg3", arg3)
        # Print objective[i]
        objective[i] = arg1 + arg2 + arg3
        #print("Print the objective at the iteration ", i, ": ", objective[i])
        # Stopping criteria
        if i>=2 and abs(objective[i] - objective[i-1]) < 10**(-4):
            break
    # Set unsignificant edges to 0
    result = prune(L, threshhold)
    
    return([result, hatS, beta, L])

vglr = np.vectorize(GSPRegression, excluded=['X_noisy', 'P'])