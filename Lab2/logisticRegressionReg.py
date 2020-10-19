import numpy as np
from sigmoid import sigmoid

def h(X, Theta):
    return sigmoid(X @ Theta)

def logisticRegressionReg(X, Y, Theta, _lambda):
    m = Y.size
    n = Theta.size
    I = np.ones([m, 1])
    J = (-Y.T @ np.log(h(X, Theta)) - (I - Y).T @ np.log(I - h(X, Theta)) + _lambda * sum(Theta[1:n]**2) / 2) / m
    _Theta = Theta.copy()
    _Theta[0, 0] = 0
    grad = (X.T @ (h(X, Theta) - Y) + _lambda * _Theta) / m
    return J, grad