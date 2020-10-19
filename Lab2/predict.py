import numpy as np
from sigmoid import sigmoid

def predict(Theta, X):
    m = X.shape[0]
    p = np.zeros([m, 1])
    for i in range(0, m):
        if(sigmoid(X[i, :] @ Theta) >= 0.5):
            p[i] = 1
    return p