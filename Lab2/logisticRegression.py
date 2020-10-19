import numpy as np
from sigmoid import sigmoid

def h(X, Theta):
    return sigmoid(X @ Theta)

def logisticRegression(X, Y, Theta):
    m = Y.size
    I = np.ones([m, 1])
    J = (-Y.T @ np.log(h(X, Theta)) - (I - Y).T @ np.log(I - h(X, Theta))) / m
    grad = (X.T @ (h(X, Theta) - Y)) / m
    return J, grad