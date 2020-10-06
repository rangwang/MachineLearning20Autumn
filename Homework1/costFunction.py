import numpy as np
import math as m1

def costFunction(x_data, y_data, theta):
    m = np.size(x_data, 0)
    J = np.sum((np.dot(x_data, theta) - y_data)**2) / (2*m) #代价函数
    grad = np.dot(x_data.T, np.dot(x_data, theta) - y_data) / m
    return J, grad

def g_costFunction(x_data, y_data, theta, p_lambda):
    m = np.size(x_data, 0)
    J = (np.sum((np.dot(x_data, theta) - y_data)**2) + p_lambda * np.sum(theta**2)) / (2*m)
    tmp = p_lambda * theta.T / m
    tmp[0] = 0
    grad = np.sum((np.dot(x_data, theta) - y_data)*x_data, axis=0) / m + tmp
    grad = grad.T 
    return J, grad