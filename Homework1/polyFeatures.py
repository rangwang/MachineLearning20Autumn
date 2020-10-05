import numpy as np 

def polyfeatures(x_data, degree):
    m = np.size(x_data, 0)
    x_poly = np.empty([m, degree+1])
    for i in range(0, m):
        for j in range(0, degree+1):
            x_poly[i, j] = np.power(x_data[i, 0], j)
    return x_poly