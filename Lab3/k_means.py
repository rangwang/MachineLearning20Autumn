import numpy as np
import random

# 初始化聚类中心，X为特征点集，k是聚类个数 
def initCentroids(X, k):
    m, n = X.shape
    randidx = random.sample(range(m), k)
    centroids = X[randidx]
    return centroids

# 根据聚类中心centroids，找到各特征点最近的中心
def findClosestCentroids(X, centroids):
    m, n = X.shape
    K = centroids.shape[0]
    idx = np.zeros([m, 1])
    for i in range(m):
        min_dis = float('inf')
        for j in range(K):
            dis = (X[i] - centroids[j]) @ (X[i] - centroids[j]).T
            if dis < min_dis:
                min_dis = dis
                idx[i] = j
    return idx

# 根据特征点集X和它们所属的聚类idx以及聚类个数k计算中心
def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros([K, n])
    for i in range(K):
        points_mean = np.zeros([1, n])
        C_k = 0
        for j in range(m):
            if idx[j] == i:
                points_mean = points_mean + X[j]
                C_k = C_k + 1
        # print(poi?)
        centroids[i] = points_mean / C_k
    return centroids