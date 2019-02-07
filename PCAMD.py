# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

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
    fig = plt.plot(mdx[:, i])
    plt.axis('off')
plt.show()
