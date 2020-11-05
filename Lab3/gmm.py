import numpy as np
import collections
from scipy.stats import multivariate_normal


# 根据数据和参数生成三组似然函数
def likelihoods(X, mu, sigma, K):
    m, n = X.shape
    Likelihoods = np.zeros([m, K])
    for i in range(K):
        Likelihoods[:, i] = multivariate_normal.pdf(X, mu[i], sigma[i])
    return Likelihoods

# 对于每组似然函数生成一个后验分布，后续比较相同数据的三组似然函数的后验分布
# 取其大者作为我们预测的类别label
def expectation(alpha, Likelihoods):
    m = Likelihoods.shape[0]
    weightedLikelihoods = alpha * Likelihoods
    sumLikelihoods = np.expand_dims(np.sum(weightedLikelihoods, axis=1), axis=1)
    gamma = weightedLikelihoods / sumLikelihoods
    idx = gamma.argmax(axis = 1).reshape(m, 1)
    return idx, gamma

def maximization(X, gamma, mu, alpha):
    m, n = X.shape
    K = mu.shape[0]
    new_mu = np.zeros([K, n])
    new_sigma = collections.defaultdict(list)
    for i in range(K):
        new_sigma[i] = np.eye(n) * 0.3
    new_alpha = np.zeros([1, K])
    for i in range(K):
        tmpgamma = np.expand_dims(gamma[:, i], axis=1)
        new_mu[i] = (tmpgamma * X).sum(axis=0) / tmpgamma.sum()
        new_sigma[i] = (X - new_mu[i]).T @ ((X - new_mu[i]) * tmpgamma) / tmpgamma.sum()
    new_alpha = gamma.sum(axis=0) / m
    return new_mu, new_sigma, new_alpha

