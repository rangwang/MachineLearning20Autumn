import numpy as np 

def g_normalEquation(x_data, y_data, p_lambda):
    n = np.size(x_data, 1) - 1
    L = np.identity(n+1)
    L[0, 0] = 0
    return np.dot(np.dot(np.linalg.pinv(np.dot(x_data.T, x_data) + p_lambda*L), x_data.T), y_data)