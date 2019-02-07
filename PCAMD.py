# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.decomposition import PCA

# Microseismic data input
mat = scio.loadmat('tensor24.mat')
MD = mat['xyztensor']
print(MD.shape)

# Separation into x, y, z components
mdx = MD[:, :, 64]
mdy = MD[:, :, 65]
mdz = MD[:, :, 66]

# Plot the data
k = mdx.shape[1]
for i in range(0, k):
    plt.subplot(k, 1, i+1)
    fig = plt.plot(mdx[:, -(i+1)])
    plt.axis('off')
plt.show()

# PCA analysis
mdx = np.array([mdx])
mdx = mdx[0, :, :]
print(mdx.shape)
pca = PCA().fit(mdx)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()

# Trying to use only 50% variance to reduce the noise
pca = PCA(0.5).fit(mdx)
print(pca.n_components_)

# Now use inverse transform to reconstruct
# filtered data

components = pca.transform(mdx)
filtered = pca.inverse_transform(components)
print(filtered.shape)

# Plot the filtered data
k = filtered.shape[1]
for i in range(0, k):
    plt.subplot(k, 1, i+1)
    fig = plt.plot(filtered[:, -(i+1)])
    plt.axis('off')
plt.show()
