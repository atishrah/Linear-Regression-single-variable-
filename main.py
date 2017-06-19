import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from ComputeCost import ComputeCost
from gradient_descent import gradient_descent 

data = pd.read_csv('ex1data1.txt',header=None,names=['pop','profit'])
data['ones']=1
Xs = data['pop']
X = data[['ones','pop']]
Y = data['profit']
n = len(X.columns)
m = len(X.index)
print(n)
plt.scatter(Xs,Y,s=10)
plt.title('population v/s profit')
#plt.show()
#print(X)
theta = np.zeros([n,1])
iterations = 1500
alpha = 0.01

initial_cost = ComputeCost(X,Y,theta)
print(initial_cost)
theta = gradient_descent(X,Y,theta,alpha,iterations)
print(theta)
Yl = np.dot(X,theta)
print(Yl)

plt.plot(Xs,Yl)

plt.show()
