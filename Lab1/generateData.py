import numpy as np 
import sys
# from matplotlib import pyplot as plt 
 
def original_func(x):
    return np.sin(2.0*np.pi*x);

def add_noise(func, sample_size, std):
    x = np.linspace(0, 1, sample_size).reshape(sample_size, 1)
    t = func(x) + np.random.normal(scale = std, size = x.shape)
    return x, t

# x_train, y_train = add_noise(original_func, 10, 0.1)
# x_origin = np.linspace(0, 1, 100)
# y_origin = original_func(x_origin)

# plt.scatter(x_train, y_train, edgecolor="g", facecolor="none", s=50, label="trainning data")
# plt.plot(x_origin, y_origin, c="b", label="$\sin(2\pi x)$") 

# plt.title("Generate $\sin(2\pi x)$ Data") 
# plt.xlabel("x axis") 
# plt.ylabel("y axis") 
# plt.legend()
# plt.show()