import numpy as np
import pandas as pd
from ComputeCost import ComputeCost 
def gradient_descent(X,Y,theta,alpha,num_iters) :
    J_history = np.zeros([num_iters,1])
    n = len(X.columns)
    m = len(X.index)
    err = np.zeros([m,1])
    for i in range(num_iters):
        h= np.dot(X,theta)
        for j in range(m):
            err[j] = h[j]-Y.iloc[j]
        temp1 = np.dot(X.T,err)
        theta = theta - (alpha*(temp1))/m
        J_history[i] = ComputeCost(X,Y,theta)

    return theta

        

        
