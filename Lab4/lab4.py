import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generatedata(size):
    mean = [4, 4, 4]
    cov = [[0.01, 0, 0], [0, 4, 0], [0, 0, 9]]
    data = np.random.multivariate_normal(mean, cov, size)
    return data

def pca(X):
    m, n = X.shape
    cov = X.T @ X / m
    U, S, V = np.linalg.svd(cov)
    return U

def featureNormalize(X):
    m, n = X.shape
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0)
    # print(X_norm)
    # X_norm = X_norm / sigma
    # print(X_norm)
    
    return X_norm, mu, sigma

def projectData(X, U, K):
    U_reduce = U[:, 0:K]
    Z = X @ U_reduce
    return Z

def recoverData(Z, U, K):
    U_reduce = U[:, 0:K]
    X_rec = Z @ U_reduce.T
    return X_rec

def displayData(X, n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

def psnr(X_origin, X_pca):
    M, N = X_origin.shape
    difference = X_origin - X_pca
    difference = difference ** 2
    mse = np.sum(difference) / (M * N)
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

# 人工生成数据
X = generatedata(100)
X_norm, mu, sigma = featureNormalize(X)
U = pca(X_norm)
Z = projectData(X_norm, U, 2)
X_rec = recoverData(Z, U, 2)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], c ='r', label='normalizaiton')
ax.scatter(X_rec[:, 0], X_rec[:, 1], X_rec[:, 2], c ='b', label='PCA')
ax.legend(loc='best')

# 人脸数据
mat = sio.loadmat('./ex7faces.mat')
X= np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get('X')])
print("人脸数据集大小：", X.shape)
X, mu, sigma = featureNormalize(X)
displayData(X, n=64)
U = pca(X)
Z = projectData(X, U, 1)
X_rec = recoverData(Z, U, 1)
displayData(X_rec, n = 64)
compression = [900, 625, 225, 100, 64, 36, 16]
for i in range(7):
    U = pca(X)
    Z = projectData(X, U, compression[i])
    X_rec = recoverData(Z, U, compression[i])
    print("信噪比：", psnr(X, X_rec))

 
# 添加坐标轴(顺序是Z, Y, X)
ax.set_ylim(ymax=4, ymin=-4)
ax.set_xlim(xmax=1, xmin=-1)
ax.set_zlim(zmax=4, zmin=-4)
ax.set_zlabel('Z', fontdict={'size': 8, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 8, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 8, 'color': 'red'})

plt.show()