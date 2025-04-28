#from numba import njit, float64
import numpy as np
import time
import cverle

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

r = r.astype(np.float64)
v = v.astype(np.float64)
m = m.astype(np.float64)
#plot_initial_distr(r, v, rmin - 5, rmax + 5, rmin - 5, rmax + 5, figsize=(4,4))
rx = r[:,0]
ry = r[:,1]
vx = v[:,0]
vy = v[:,1]

t = np.arange(0, 100, 1)
#t = np.arange(0, 5*31.536*10**6, 86400/5)
t = t.astype(np.float64)
tstart = time.time()
res = cverle.solve_ode_c(N, rx, ry, vx, vy, m, t)
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
# np.save('./sol_cverle_px.npy', thin_positions_x)
# np.save('./sol_cverle_py.npy', thin_positions_y)
# np.save('./sol_cverle_vx.npy', thin_velocities_x)
# np.save('./sol_cverle_vy.npy', thin_velocities_y)
