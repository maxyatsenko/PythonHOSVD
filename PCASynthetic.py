# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.decomposition import PCA

# Synthetic data input
mat = scio.loadmat('synthTensor.mat')
MD = mat['eventtensor']
MD = np.array([MD])
MD = MD[0, :, :, :]
print(MD.shape)

# Separation into x, y, z components
md = MD[:, :, 0]
# mdy = MD[:, :, 65]
# mdz = MD[:, :, 66]

# Plot the data
k = md.shape[1]
for i in range(0, k):
    plt.subplot(k, 1, i+1)
    fig = plt.plot(md[:, -(i+1)])
    plt.axis('off')
plt.show()

# PCA analysis
pca = PCA().fit(md)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()

# Now let's add some random noise
np.random.seed(42)
noisy = np.random.normal(md, 0.0001)
# Plot the data
k = noisy.shape[1]
for i in range(0, k):
    plt.subplot(k, 1, i+1)
    fig = plt.plot(noisy[:, -(i+1)])
    plt.axis('off')
plt.show()


# Trying to use only 50% variance to reduce the noise
pca = PCA(0.9).fit(noisy)
print(pca.n_components_)

# Now use inverse transform to reconstruct
# filtered data

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
print(filtered.shape)

# Plot the filtered data
k = filtered.shape[1]
for i in range(0, k):
    plt.subplot(k, 1, i+1)
    fig = plt.plot(filtered[:, -(i+1)])
    plt.axis('off')
plt.show()
