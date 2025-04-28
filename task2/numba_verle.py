# import os
# # note that this must be executed before 'import numba'
# os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
from numba import njit, float64
import numpy as np
import time

N = 400

def thinout_array(array, start, end, step):
  new_size = (end - start)//step + 1
  if array.ndim == 1:
    new_array = np.zeros(new_size)
    for i in range(new_size):
      new_array[i] = array[start + i*step]
  else:
    new_array = np.zeros((array.shape[0], new_size))
    for i in range(new_size):
      new_array[:,i] = array[:, start + i*step]
  return new_array

@njit(float64[:, :](float64[:], float64[:], float64[:]), parallel=True)
def calc_acceleration_fast(rx, ry, m):
  #6.6743 × 10-11 м3 кг-1 с-2
  G = 6.6743 * 10**(-11)
  ax = np.zeros(N, dtype = float64)
  ay = np.zeros(N, dtype = float64)
  #print('calc acceleration')
  for i in range(N):
    for j in range(N):
      if i == j:
        continue
      if rx[i] == ry[j] and ry[i]==ry[j]:
        continue
      ax[i] = ax[i] + m[j]*(rx[j] - rx[i])/(((rx[j] - rx[i])**2 + (ry[j] - ry[i])**2)**0.5)**3
      ay[i] = ay[i] + m[j]*(ry[j] - ry[i])/(((rx[j] - rx[i])**2 + (ry[j] - ry[i])**2)**0.5)**3
  ax = ax*G
  ay = ay*G
  #res = np.concatenate(ax, ay, dtype=float64)
  return np.vstack((ax, ay))

@njit(float64[:,:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64[:]), parallel=True)
def Verlet_fast(rx, ry, vx, vy, ax, ay, dt, m):
  rx_n1 = rx + vx*dt + ax/2*dt**2
  ry_n1 = ry + vy*dt + ay/2*dt**2

  ax_n1, ay_n1 = calc_acceleration_fast(rx_n1, ry_n1, m)
  vx_n1 = vx + 1/2*(ax+ax_n1)*dt
  vy_n1 = vy + 1/2*(ay+ay_n1)*dt
  return np.vstack((rx_n1, ry_n1, vx_n1, vy_n1, ax_n1, ay_n1))
  
@njit(float64[:,:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), parallel=True)
def solve_ode_fast(rx0, ry0, vx0, vy0, m, t):
  rx = np.zeros((N, t.shape[0]), dtype = float64)
  ry = np.zeros((N, t.shape[0]), dtype = float64)
  vx = np.zeros((N, t.shape[0]), dtype = float64)
  vy = np.zeros((N, t.shape[0]), dtype = float64)
  ax, ay = calc_acceleration_fast(rx0, ry0, m)
  rx[:,0] = rx0
  ry[:,0] = ry0
  vx[:,0] = vx0
  vy[:,0] = vy0
  for i in range(1, t.shape[0]):
    rx_slice = rx[:,i-1]
    ry_slice = ry[:,i-1]
    vx_slice = vx[:,i-1]
    vy_slice = vy[:,i-1]
    rx[:,i], ry[:,i], vx[:,i], vy[:,i], ax, ay = Verlet_fast(rx_slice, ry_slice, vx_slice, vy_slice,
                                                                              ax, ay, t[i] - t[i-1], m)
  return np.vstack((rx, ry, vx, vy))

rmin = 0
rmax = 300
vmin = -5
vmax = 5
mmin = 1
mmax = 5000000

N = 400
#r, v, m = rand_initial_values(N, rmin, rmax, vmin, vmax, mmin, mmax)
r = np.load('r_400.npy') 
v = np.load('v_400.npy')
m = np.load('m_400.npy')

# r = r.astype(np.float64)
# v = v.astype(np.float64)
# m = m.astype(np.float64)

#Solar system
# cx = 5000*10**3
# cy = 5000*10**3
# a = 4514.953*10**3
# e = 0.0097
# b = a * (1 - e**2)**0.5
# Sx = cx + a*e
# Sy = cy
# r = np.array([
#     [Sx, Sy], #Sun
#     [Sx + 45.9*10**3, Sy], #Mercury
#     [Sx + 107.476*10**3, Sy], #Venus
#     [Sx + 47.098*10**3, Sy],#Earth
#     [Sx + 206.665*10**3, Sy],#Mars
#     [Sx + 740.595*10**3, Sy],#Jupiter
#     [Sx + 1357.554*10**3, Sy],#Saturne
#     [Sx + 2732.696*10**3, Sy],#Uranus
#     [Sx + 4471.050*10**3, Sy] #Neptune
#     ], dtype = np.float64)
# v = np.array([
#     [0,0], #Sun
#     [0, 0.0566], #Mercury
#     [0,0.035], #Venus
#     [0,0.03],#Earth
#     [0,0.0265],#Mars
#     [0,0.01372],#Jupiter
#     [0,0.01014],#Saturne
#     [0,0.00713],#Uranus
#     [0,0.00547 ] #Neptune
#     ], dtype = np.float64)

# m = np.array([
#     199*10**10, #Sun
#     0.33*10**6, #Mercury
#     4.87*10**6, #Venus
#     5.97*10**6,#Earth
#     0.642*10**6,#Mars
#     1898*10**6,#Jupiter
#     568*10**6,#Saturne
#     86.8*10**6,#Uranus
#     102*10**6 #Neptune
# ], dtype = np.float64)

#plot_initial_distr(r, v, rmin - 5, rmax + 5, rmin - 5, rmax + 5, figsize=(4,4))
# time.sleep(2)
rx = r[:,0]
ry = r[:,1]
vx = v[:,0]
vy = v[:,1]

t = np.arange(0, 100, 1)
#t = np.arange(0, 5*31.536*10**6, 86400/5)
t = t.astype(np.float64)
tstart = time.time()
res = solve_ode_fast(rx, ry, vx, vy, m, t)
tend = time.time()

positions_x = res[0:N]
positions_y = res[N:2*N]
velocities_x = res[2*N:3*N]
velocities_y = res[3*N:]
print(tend - tstart)

# thin_t = thinout_array(t, 0, t.shape[0] - 1, step = 25)
# thin_positions_x = thinout_array(positions_x, 0, t.shape[0] - 1, step = 25)
# thin_positions_y = thinout_array(positions_y, 0, t.shape[0] - 1, step = 25)
# thin_velocities_x = thinout_array(velocities_x, 0, t.shape[0] - 1, step = 25)
# thin_velocities_y = thinout_array(velocities_y, 0, t.shape[0] - 1, step = 25)
# np.save('./sol_numverle_px.npy', thin_positions_x)
# np.save('./sol_numverle_py.npy', thin_positions_y)
# np.save('./sol_numverle_vx.npy', thin_velocities_x)
# np.save('./sol_numverle_vy.npy', thin_velocities_y)

