import numpy as np
import collections
from matplotlib import pyplot as plt
from generateData import generateData
from k_means import initCentroids
from k_means import computeCentroids
from k_means import findClosestCentroids
from gmm import likelihoods
from gmm import expectation
from gmm import maximization
from UCIdataset import readUCIdata
from UCIdataset import accuracy

# 初始化样本数据参数
mu_0 = np.array([1, 1])
mu_1 = np.array([7, 1])
mu_2 = np.array([4, 4])
mu = np.array([[1, 1], [7, 1], [4, 4]])
sigma0 = sigma1 = sigma2 = 0.3*np.eye(2)
sigma = collections.defaultdict(list)
for i in range(3):
    sigma[i] = np.eye(2) * 0.3

train_size = 20
delta = 1e-7

# 生成样本数据
train_Data0 = generateData(mu_0, sigma0, train_size, 0)
train_Data1 = generateData(mu_1, sigma1, train_size, 1)
train_Data2 = generateData(mu_2, sigma2, train_size, 2)

# 三个聚类整合
tmpX = np.r_[train_Data0, train_Data1]
X_train = np.r_[tmpX, train_Data2]
X = X_train[:, 0:2]
Y = X_train[:, 2]
m, n = X.shape


# 通过100次循环选出最佳的初始点对
init_Centroids = np.zeros([3, n])
min_cost = float('inf')
for i in range(100):
    cost = 0
    centroids = initCentroids(X, 3)
    idx = findClosestCentroids(X, centroids)
    for j in range(m):
        cost = cost + (X[j] - centroids[int(idx[j])]) @ (X[j] - centroids[int(idx[j])]).T
    if cost < min_cost:
        min_cost = cost
        init_Centroids = centroids  

mu = init_Centroids
'''==============================K-means算法================================'''   
# 根据聚类中心初始化点集中所有点所属类别
idx = findClosestCentroids(X, init_Centroids)

# 迭代找出聚类中心 
i = 0  
while (True):
    centroids = computeCentroids(X, idx, 3)
    if i > 1000:
        break
    if np.sum((centroids - init_Centroids)**2) < delta:
        break
    else:
        i = i + 1
        init_Centroids = centroids
        idx = findClosestCentroids(X, centroids)

print("迭代了%d次" %i)
print("K-means生成的聚类中心为\n", centroids)

cluster_Data0 = np.ones([0, 2])
cluster_Data1 = np.ones([0, 2])
cluster_Data2 = np.ones([0, 2])
for i in range(train_size*3):
    if idx[i] == 0:
        cluster_Data0 = np.r_[cluster_Data0, X[i].reshape(1,2)]
    elif idx[i] == 1:
        cluster_Data1 = np.r_[cluster_Data1, X[i].reshape(1,2)]
    elif idx[i] == 2:
        cluster_Data2 = np.r_[cluster_Data2, X[i].reshape(1,2)]

plt.subplot(221)
plt.scatter(train_Data0[:, 0], train_Data0[:, 1])
plt.scatter(train_Data1[:, 0], train_Data1[:, 1])
plt.scatter(train_Data2[:, 0], train_Data2[:, 1])
plt.scatter(np.array([1,4,7]), np.array([1,4,1]))

plt.subplot(222)
plt.scatter(cluster_Data0[:, 0], cluster_Data0[:, 1])
plt.scatter(cluster_Data1[:, 0], cluster_Data1[:, 1])
plt.scatter(cluster_Data2[:, 0], cluster_Data2[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1])

'''==============================GMM-EM算法================================'''
# print(sigma)

alpha_0 = 1.0/3 * np.ones(3).reshape(1, 3)

i = 0
while(True):
    Likelihoods_0 = likelihoods(X, mu, sigma, 3)
    idx_1, gamma = expectation(alpha_0, Likelihoods_0)
    new_mu, new_sigma, new_alpha = maximization(X, gamma, mu, alpha_0)
    if i > 1000:
        break
    if(np.linalg.norm(new_mu - mu)  + np.linalg.norm(new_alpha - alpha_0) \
        + np.sum([np.linalg.norm(sigma[i] - new_sigma[i]) for i in range(3)])< delta):
        break
    else:
        i = i + 1
        mu = new_mu
        alpha_0 = new_alpha
        sigma = new_sigma

idx_1, gamma = expectation(new_alpha, Likelihoods_0)

print("迭代了%d次" %i)

cluster_Data0 = np.ones([0, 2])
cluster_Data1 = np.ones([0, 2])
cluster_Data2 = np.ones([0, 2])

for i in range(train_size*3):
    if idx_1[i] == 0:
        cluster_Data0 = np.r_[cluster_Data0, X[i].reshape(1,2)]
    elif idx_1[i] == 1:
        cluster_Data1 = np.r_[cluster_Data1, X[i].reshape(1,2)]
    elif idx_1[i] == 2:
        cluster_Data2 = np.r_[cluster_Data2, X[i].reshape(1,2)]

plt.subplot(223)
plt.scatter(cluster_Data0[:, 0], cluster_Data0[:, 1])
plt.scatter(cluster_Data1[:, 0], cluster_Data1[:, 1])
plt.scatter(cluster_Data2[:, 0], cluster_Data2[:, 1])

X, Y = readUCIdata('iris.data')
m, n = X.shape
'''===========================K-means算法UCI数据集=============================''' 
init_Centroids = np.zeros([3, n])
min_cost = float('inf')
for i in range(100):
    cost = 0
    centroids = initCentroids(X, 3)
    idx = findClosestCentroids(X, centroids)
    for j in range(m):
        cost = cost + (X[j] - centroids[int(idx[j])]) @ (X[j] - centroids[int(idx[j])]).T
    if cost < min_cost:
        min_cost = cost
        init_Centroids = centroids  
mu = init_Centroids

# 根据聚类中心初始化点集中所有点所属类别
idx = findClosestCentroids(X, init_Centroids)

# 迭代找出聚类中心 
i = 0  
while (True):
    centroids = computeCentroids(X, idx, 3)
    if np.sum((centroids - init_Centroids)**2) < delta:
        break
    else:
        i = i + 1
        init_Centroids = centroids
        idx = findClosestCentroids(X, centroids)

print("K-Means迭代了%d次" %i)
print("K-means生成的聚类中心为\n", centroids)
print("K-Means准确率为", accuracy(idx, Y))

'''===========================GMM-EM算法UCI数据集==============================''' 
alpha_0 = 1.0/3 * np.ones(3).reshape(1, 3)
sigma = collections.defaultdict(list)
for i in range(3):
    sigma[i] = np.eye(4) * 0.3

i = 0
while(True):
    Likelihoods_0 = likelihoods(X, mu, sigma, 3)
    idx_1, gamma = expectation(alpha_0, Likelihoods_0)
    new_mu, new_sigma, new_alpha = maximization(X, gamma, mu, alpha_0)
    if i > 1000:
        break
    if(np.linalg.norm(new_mu - mu)  + np.linalg.norm(new_alpha - alpha_0) \
        + np.sum([np.linalg.norm(sigma[i] - new_sigma[i]) for i in range(3)])< delta):
        break
    else:
        i = i + 1
        mu = new_mu
        alpha_0 = new_alpha
        sigma = new_sigma

idx_1, gamma = expectation(new_alpha, Likelihoods_0)

print("GMM-EM迭代了%d次" %i)
print("GMM-EM准确率为", accuracy(idx_1, Y))
plt.show()
