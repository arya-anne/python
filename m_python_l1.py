import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
#import skimage.io

n = 10
A = np.diag(np.linspace(0, n - 1, n, dtype = int))
A = np.fliplr(A)
xr = np.roll(A,1,axis=1)
print(xr)
pp = 1

# points_cnt = 100
# a = 4
# b = 3
# x, y = np.meshgrid( np.linspace(-math.pi, math.pi, points_cnt, dtype = float), np.linspace(-math.pi, math.pi, points_cnt, dtype = float))
# z = np.sin(a*x)*np.sin(b*y)
# #z3 = np.multiply(np.sin(a*x), np.sin(b*y))
#
# fig = plt.figure()
# ax = fig.add_subplot(211)
# #ax.imshow(z, interpolation='nearest')
# skimage.io.imshow(z)
# plt.show()



# n = 10
# A = np.zeros([n, n], dtype=int)
# #A = np.fill_diagonal(np.fliplr(A), np.linspace(0, 9, 1), offset=-1)
# np.fill_diagonal(np.fliplr(A), np.linspace(0, n, n+1, dtype = int))
# xr = np.roll(A,1,axis=1)
# print(xr)



