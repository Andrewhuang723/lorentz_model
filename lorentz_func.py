import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

T = 10
data_size = 100000
t = np.linspace(0, T, data_size)
b = 8 / 3
sig = 10
r = 28


def Lorenz(state, t):
    x, y, z = state  # Unpack the state vector
    return sig * (y - x), x * (r - z) - y, x * y - b * z  # Derivatives


data = []
fig = plt.figure()
for i in range(10):
    x0 = 30 * (np.random.rand(3) - 0.5)
    states = odeint(Lorenz, x0, t)
    ax = fig.gca(projection="3d")
    ax.plot3D(states[:, 0], states[:, 1], states[:, 2])
    data.append(states)
plt.show()
data = np.asarray(data)
print(data.shape)