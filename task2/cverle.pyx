import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# distributed with Numpy).
# Here we've used the name "cnp" to make it easier to understand what
# comes from the cimported module and what comes from the imported module,
# however you can use the same name for both if you wish.
cimport numpy as cnp

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cnp.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.

ctypedef cnp.float64_t DTYPE_t

from cython.parallel import prange
#cimport openmp
from libc.math cimport sqrt

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calc_acceleration_c(int N, cnp.ndarray[DTYPE_t, ndim=1] rx, 
                                  cnp.ndarray[DTYPE_t, ndim=1] ry, cnp.ndarray[DTYPE_t, ndim=1] m):
  #6.6743 × 10-11 м3 кг-1 с-2
  cdef double G = 6.6743 * 10**(-11)
  cdef cnp.ndarray[DTYPE_t, ndim=1] ax = np.zeros(N, dtype = DTYPE)
  cdef cnp.ndarray[DTYPE_t, ndim=1] ay = np.zeros(N, dtype = DTYPE)
  cdef int i, j
  #for i in range(N):
  for i in prange(N, nogil=True):
    #valx = 0
    #valy = 0
    for j in range(N):
      if i == j:
        continue
      if rx[i] == ry[j] and ry[i]==ry[j]:
        continue
      ax[i] = ax[i] + m[j]*(rx[j] - rx[i])/(sqrt((rx[j] - rx[i])**2 + (ry[j] - ry[i])**2))**3
      ay[i] = ay[i] + m[j]*(ry[j] - ry[i])/(sqrt((rx[j] - rx[i])**2 + (ry[j] - ry[i])**2))**3
  ax = ax*G
  ay = ay*G
  cdef cnp.ndarray[DTYPE_t, ndim=2] res = np.vstack((ax, ay))
  #res = np.concatenate(ax, ay, dtype=float64)
  return res

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def Verlet_c(int N, cnp.ndarray[DTYPE_t, ndim=1] rx, cnp.ndarray[DTYPE_t, ndim=1] ry,
                       cnp.ndarray[DTYPE_t, ndim=1] vx, cnp.ndarray[DTYPE_t, ndim=1] vy, 
                       cnp.ndarray[DTYPE_t, ndim=1] ax, cnp.ndarray[DTYPE_t, ndim=1] ay, 
                       double dt, cnp.ndarray[DTYPE_t, ndim=1] m):
  cdef cnp.ndarray[DTYPE_t, ndim=1]  rx_n1 = rx + vx*dt + ax/2*dt**2
  cdef cnp.ndarray[DTYPE_t, ndim=1]  ry_n1 = ry + vy*dt + ay/2*dt**2

  cdef cnp.ndarray[DTYPE_t, ndim=1] ax_n1, ay_n1
  ax_n1, ay_n1 = calc_acceleration_c(N, rx_n1, ry_n1, m)

  cdef cnp.ndarray[DTYPE_t, ndim=1] vx_n1 = vx + 1/2*(ax+ax_n1)*dt
  cdef cnp.ndarray[DTYPE_t, ndim=1] vy_n1 = vy + 1/2*(ay+ay_n1)*dt
  cdef cnp.ndarray[DTYPE_t, ndim=2] res = np.vstack((rx_n1, ry_n1, vx_n1, vy_n1, ax_n1, ay_n1))
  return res

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def solve_ode_c(int N, cnp.ndarray[DTYPE_t, ndim=1] rx0, cnp.ndarray[DTYPE_t, ndim=1] ry0,
                       cnp.ndarray[DTYPE_t, ndim=1] vx0, cnp.ndarray[DTYPE_t, ndim=1] vy0,
                       cnp.ndarray[DTYPE_t, ndim=1] m, cnp.ndarray[DTYPE_t, ndim=1] t):
  cdef cnp.ndarray[DTYPE_t, ndim=2] rx = np.zeros((N, t.shape[0]), dtype = DTYPE)
  cdef cnp.ndarray[DTYPE_t, ndim=2] ry = np.zeros((N, t.shape[0]), dtype = DTYPE)
  cdef cnp.ndarray[DTYPE_t, ndim=2] vx = np.zeros((N, t.shape[0]), dtype = DTYPE)
  cdef cnp.ndarray[DTYPE_t, ndim=2] vy = np.zeros((N, t.shape[0]), dtype = DTYPE)

  cdef cnp.ndarray[DTYPE_t, ndim=1] ax, ay
  ax, ay = calc_acceleration_c(N, rx0, ry0, m)
  rx[:,0] = rx0
  ry[:,0] = ry0
  vx[:,0] = vx0
  vy[:,0] = vy0
  cdef int i
  for i in range(1, t.shape[0]):
    #cdef cnp.ndarray[DTYPE_t, ndim=1] rx_slice = rx[:,i-1]
    #cdef cnp.ndarray[DTYPE_t, ndim=1] ry_slice = ry[:,i-1]
    #cdef cnp.ndarray[DTYPE_t, ndim=1] vx_slice = vx[:,i-1]
    #cdef cnp.ndarray[DTYPE_t, ndim=1] vy_slice = vy[:,i-1]
    #rx[:,i], ry[:,i], vx[:,i], vy[:,i], ax, ay = Verlet_c(N, rx_slice, ry_slice, vx_slice, vy_slice,
                                                                             # ax, ay, t[i] - t[i-1], m)
    rx[:,i], ry[:,i], vx[:,i], vy[:,i], ax, ay = Verlet_c(N, rx[:,i-1], ry[:,i-1], vx[:,i-1], vy[:,i-1],
                                                                              ax, ay, t[i] - t[i-1], m)
                                                                              
  cdef cnp.ndarray[DTYPE_t, ndim=2] res = np.vstack((rx, ry, vx, vy))
  return res