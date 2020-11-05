import numpy as np
import pandas as pd
import itertools

classes = list(itertools.permutations([0, 1, 2], 3))
def readUCIdata(setName):
    dataset = pd.read_csv(setName)
    tmpX = dataset.drop('class', axis = 1)
    tmpY = dataset['class']
    X = np.array(tmpX)
    Y = np.array(tmpY)
    m = Y.shape[0]
    # 对标签和数字作映射
    for i in range(m):
        if Y[i] == "Iris-setosa":
            Y[i] = 0
        if Y[i] == "Iris-versicolor":
            Y[i] = 1
        if Y[i] == "Iris-virginica":
            Y[i] = 2
    return X, Y

def accuracy(idx, Y):
    m = Y.shape[0]
    counts = []
    for i in range(len(classes)):
        count = 0
        # 对于所有的排列可能进行搜索，选取counts最大的组合
        for j in range(m):
            if idx[j] == classes[i][Y[j]]:
                count += 1
        counts.append(count)
    return np.max(counts) * 1.0 / m

X, Y = readUCIdata('iris.data')
print(Y.size)