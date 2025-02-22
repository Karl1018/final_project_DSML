# Load npy file and plot the data in 3D
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/lorenz63_test.npy')
print(data.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], marker='.', s=1)

plt.show()
