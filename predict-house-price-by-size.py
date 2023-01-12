# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 06:07:07 2023

@author: Scadra
"""

import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_cost_function(x, y, w, b): 
    """
    Calculate the cost function for linear regression.
    The difference between the target (the value from our test) and the prediction is calculated and squared.
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        #f_wb is a prediction calculated
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost


#size in 1000 square feet
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
#price in 1000 square in dollar
y_train = np.array([250, 290, 475,  430, 622, 720,])


# Initialize model parameters
w = 0
b = 0

# Compute the cost
cost = compute_cost_function(x_train, y_train, w, b)
print("Initial cost: ", cost)


# Create a grid of (w,b) values
ws = np.linspace(-50, 50, 100)
bs = np.linspace(-100000, 100000, 100)
W, B = np.meshgrid(ws, bs)

# Compute the cost for all (w,b) values
Z = np.array([compute_cost_function(x_train, y_train, w_i, b_i) for w_i, b_i in zip(np.ravel(W), np.ravel(B))])

# Reshape the cost array to match the shape of (w,b)
Z = Z.reshape(W.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Cost")
ax.plot_surface(W, B, Z)
plt.show()