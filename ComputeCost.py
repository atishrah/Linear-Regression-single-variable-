import numpy as np
import pandas as pd
def ComputeCost(X,Y,theta):
    J =0
    m = len(X.index)
    h = np.dot(X,theta)
    delta = 0
    for i in range(m):
        delta = delta + ((h[i]-Y.iloc[i])**2)
    J = delta/(2*m)
    return J

    
    
	
