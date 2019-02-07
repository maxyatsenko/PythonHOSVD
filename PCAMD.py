# Imports
import numpy as np
import matplotlib.pyplot as plt

# Microseismic data input
MD = np.loadtxt('145.txt')
MD = MD.reshape(2000, 36)

# Get only x component
y = 1
mdx = np.zeros([MD.shape[0], int(MD.shape[1]/3)])
for i in range(0, int(MD.shape[1]/3)):
    mdx[:, i] = MD[:, y]
    y = y+3

# Plot the data
k = mdx.shape[1]
plt.figure(figsize=(10,15))
for i in range(0, k):
    plt.subplot(k, 1, i+1)
    plt.plot(mdx[:, i], '.-')
    plt.axis('off')
plt.show()
