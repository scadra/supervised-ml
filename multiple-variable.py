# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:59:25 2023

@author: Scadra
"""

import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[3000, 2, 230, 5], [2000, 5, 0, 2], [1000, 1, 25, 1]])
y_train = np.array([460, 178, 215])

# data is stored in numpy matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predict(x, w, b):
    """
    Predict the output of a model given input features and model weights.
    
    Parameters:
        x (np.array): Input features of shape (n_samples, n_features)
        w (np.array): Model weights of shape (n_features, )
        b (float): Model bias term 
    
    Returns:
        np.array: Model output of shape (n_samples, )
    """
    p = np.dot(x, w) + b     
    return p


x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

"""
The above code snippet extracts the first row of the X_train matrix, which corresponds to the first sample of the training dataset. The shape of x_vec and its value are printed out for verification.
"""

f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

"""
The above code snippet uses the predict function to obtain the model's prediction on the first sample of the training dataset using the initial weight and bias parameters (w_init and b_init). The shape and value of the prediction are printed out for verification.
"""

def compute_cost(X, y, w, b):
    """
    Compute the cost of a linear regression model.
    
    Parameters:
        X (np.array): Input features of shape (n_samples, n_features)
        y (np.array): Target values of shape (n_samples, )
        w (np.array): Model weights of shape (n_features, )
        b (float): Model bias term
    
    Returns:
        float: The computed cost of the linear regression model.
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b          
        cost = cost + (f_wb_i - y[i])**2   
    cost = cost / (2 * m)                   
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost w : {cost}')

def compute_gradient(X, y, w, b):
    """
    Compute the gradient of the cost function of a linear regression model.
    
    Parameters:
        X (np.array): Input features of shape (n_samples, n_features)
        y (np.array): Target values of shape (n_samples, )
        w (np.array): Model weights of shape (n_features, )
        b (float): Model bias term
    
    Returns:
        tuple: The computed gradients of the cost function with respect to the bias term (dj_db) and the weights (dj_dw)
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Perform gradient descent to optimize the weights and bias of a linear regression model.
    
    Parameters:
        X (np.array): Input features of shape (n_samples, n_features)
        y (np.array): Target values of shape (n_samples, )
        w_in (np.array): Initial model weights of shape (n_features, )
        b_in (float): Initial model bias term
        cost_function (function): Function to compute the cost of the model
        gradient_function (function): Function to compute the gradients of the cost function
        alpha (float): Learning rate
        num_iters (int): Number of iterations to perform gradient descent
        
    Returns:
        tuple: The optimized weights and bias term of the model, along with the history of the cost function during the optimization process
    """
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        dj_db,dj_dw = gradient_function(X, y, w, b)   

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

"""
The above code snippet initializes the model parameters with initial_w being an array of zeroes of the same shape as w_init and initial_b being 0. The number of iterations and the learning rate alpha are set. Then gradient_descent function is called with the training data, initial weight and bias, cost and gradient functions, learning rate and number of iterations as inputs. The final optimized weight and bias and the history of cost during the optimization process are returned. The final weight and bias and the predictions of the model on the training data are printed.
"""
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
