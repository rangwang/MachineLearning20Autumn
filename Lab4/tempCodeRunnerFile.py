mat = sio.loadmat('./ex7faces.mat')
# X = np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get('X')])
# print("人脸数据集大小：", X.shape)
# displayData(X, n=64)
# U = pca(X)
# Z = projectData(X, U, 100)
# X_rec = recoverData(Z, U, 100)
# displayData(X_rec, n = 64)
# compression = [1000, 500, 250, 125, 60, 30, 15]
# for i in range(7):
#     U = pca(X)
#     Z = projectData(X, U, compression[i])
#     X_rec = recoverData(Z, U, compression[i])
#     print("信噪比：", psnr(X, X_rec))
