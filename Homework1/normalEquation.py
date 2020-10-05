import numpy as np 

def normalEquation(x_data, y_data):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(x_data.T, x_data)), x_data.T), y_data)
    return theta