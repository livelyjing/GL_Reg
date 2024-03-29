import numpy as np
import networkx as nx
import GL_Reg.gl_reg as glr
import GL_Reg.eval_metrics as met

def findHParams(X_noisy,              # The Observed Signal
                P0,                   # The Observed Regressors
                L0,                   # The ground truth laplacian
                x_max = 1,            # center x coord of mesh
                y_max = 1,            # center y coord of mesh
                epsilon = .9,         # Distance from midpoint to edge of the mesh
                steps = 3,            # Number of recursive steps
                maxiter = 3,          # Max number of iterations GL_Reg will perform to learn a laplacian
                threshold = 10**(-3), # Threshold for pruning insignificant edges
                digits = 4,            # How many decimals to keep in coordinates/metric
                verbose = False):     # Set True to see the path grid search takes 
    # Initializations
    bestResult = glr.GSPRegression(X_noisy, P0, x_max,y_max, maxiter, threshold)
    maxMeasure = round(met.metricsprf(L0, bestResult[0])[2], digits)
    if verbose: print("Initial F-Measure is " + str(maxMeasure))
    
    # Create the first 
    X = [.01] + [round(x, digits) for x in np.linspace(x_max - epsilon, x_max + epsilon, 5) if x > 0]
    Y = [.01] + [round(y, digits) for y in np.linspace(y_max - epsilon, y_max + epsilon, 5) if y > 0]
    for i in range(steps):
        for x in X:
            for y in Y:
                curRes = glr.GSPRegression(X_noisy, P0, x,y, maxiter, threshold)
                curL = curRes[0]
                curMeasure = round(met.metricsprf(L0, curL)[2], digits)
                if verbose: print("Current F-Measure is " + str(curMeasure) + " for x = " + str(x) + " y = " + str(y))
                if curMeasure > maxMeasure:
                    x_max = x
                    y_max = y 
                    bestResult = curRes
                    maxMeasure = curMeasure
                    if verbose: print("New max of " + str(maxMeasure) + " at (" + str(x_max) + ", " + str(y_max) +")")
        if verbose: print("Step " + str(i+1) + " completed.")
        X = [round(i, digits) for i in np.linspace(x_max - epsilon*(.1), x_max + epsilon*(.1), 5) if i > 0]  
        Y = [round(j, digits) for j in np.linspace(y_max - epsilon*(.1), y_max + epsilon*(.1), 5) if j > 0]
    print("-----------------------------------------")
    print("              Result                     ")
    print("-----------------------------------------")
    print("F-Measure = " + str(maxMeasure) + " at (" + str(x_max) + ", " + str(y_max) + ")")

    return([x_max, y_max] + bestResult)
