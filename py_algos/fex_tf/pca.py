import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# generate some sample data
X = np.load('fm.npy')
y = np.load('labels.npy')

X = X[:, 2:]

# create a PCA object and fit the data
pca = PCA(n_components = 5)
pca.fit(X)

# transform the data to the new coordinate system
X_transformed = pca.transform(X)

print("Original data:\n", X)
print("Transformed data:\n", X_transformed)
print("Explained variance ratio:", pca.explained_variance_ratio_)

xx = X_transformed
for i in range(X.shape[0]):
    if y[i] == 1:
        plt.plot(xx[i, 0], xx[i, 1], 'go')
    if y[i] == 2:
        plt.plot(xx[i, 0], xx[i, 1], 'rx')

plt.show()
