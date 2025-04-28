import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

def rand_initial_values(N, rmin, rmax, vmin, vmax, mmin, mmax):
  r = rmin + (rmax - rmin)*np.random.rand(N, 2)
  v = vmin + (vmax - vmin)*np.random.rand(N, 2)
  m = mmin + (mmax - mmin)*np.random.rand(N)
  return r, v, m

#finds power of 2 equal to or greater than a given non-negative integer
def shift_bit_length(x):
    return 1<<(x-1).bit_length()

def calc_acceleration(N, rx, ry, m):
  blsize = int( shift_bit_length(N) )
  N = np.uint32(N)
  ax = np.zeros((N), dtype=np.float64)
  ay = np.zeros((N), dtype=np.float64)

  rx_gpu = cuda.mem_alloc(rx.size * rx.dtype.itemsize)
  ry_gpu = cuda.mem_alloc(ry.size * ry.dtype.itemsize)
  m_gpu =  cuda.mem_alloc(m.size * m.dtype.itemsize)
  ax_gpu = cuda.mem_alloc(ax.size * ax.dtype.itemsize)
  ay_gpu = cuda.mem_alloc(ay.size * ay.dtype.itemsize)
  datax_gpu = cuda.mem_alloc(blsize* int(N) * ax.dtype.itemsize)
  datay_gpu = cuda.mem_alloc(blsize* int(N) * ax.dtype.itemsize)

  cuda.memcpy_htod(rx_gpu, rx)
  cuda.memcpy_htod(ry_gpu, ry)
  cuda.memcpy_htod(m_gpu, m)

  func = mod.get_function("calc_accelerationGPU")
  func(rx_gpu, ry_gpu, m_gpu, datax_gpu, datay_gpu, ax_gpu, ay_gpu, N, \
      block=(blsize, 1, 1), \
      grid=(int(N), 1), shared=0)

  cuda.memcpy_dtoh(ax, ax_gpu)
  cuda.memcpy_dtoh(ay, ay_gpu)
  return ax, ay

def VerletGPU(N, rx, ry, vx, vy, ax, ay, dt, m):
  rx_n1 = rx + vx*dt + ax/2*dt**2
  ry_n1 = ry + vy*dt + ay/2*dt**2

  ax_n1, ay_n1 = calc_acceleration(N, rx_n1, ry_n1, m)
  vx_n1 = vx + 1/2*(ax+ax_n1)*dt
  vy_n1 = vy + 1/2*(ay+ay_n1)*dt
  return rx_n1, ry_n1, vx_n1, vy_n1, ax_n1, ay_n1

def solve_odeGPU(N, rx0, ry0, vx0, vy0, m, t):
  rx = np.zeros((N, t.shape[0]), dtype=np.float64)
  ry = np.zeros((N, t.shape[0]), dtype=np.float64)
  vx = np.zeros((N, t.shape[0]), dtype=np.float64)
  vy = np.zeros((N, t.shape[0]), dtype=np.float64)

  #part with GPU
  ax, ay = calc_acceleration(N, rx0, ry0, m)

  rx[:,0] = rx0
  ry[:,0] = ry0
  vx[:,0] = vx0
  vy[:,0] = vy0
  for i in range(1, t.shape[0]):
    rx[:,i], ry[:,i], vx[:,i], vy[:,i], ax, ay = VerletGPU(N, rx[:,i-1], ry[:,i-1], vx[:,i-1], vy[:,i-1],
                                                                              ax, ay, t[i] - t[i-1], m)
  return rx, ry, vx, vy

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

N = 9
#Solar system
cx = 5000*10**3
cy = 5000*10**3
a = 4514.953*10**3
e = 0.0097
b = a * (1 - e**2)**0.5
Sx = cx + a*e
Sy = cy
r = np.array([
    [Sx, Sy], #Sun
    [Sx + 45.9*10**3, Sy], #Mercury
    [Sx + 107.476*10**3, Sy], #Venus
    [Sx + 47.098*10**3, Sy],#Earth
    [Sx + 206.665*10**3, Sy],#Mars
    [Sx + 740.595*10**3, Sy],#Jupiter
    [Sx + 1357.554*10**3, Sy],#Saturne
    [Sx + 2732.696*10**3, Sy],#Uranus
    [Sx + 4471.050*10**3, Sy] #Neptune
    ], dtype = np.float64)
v = np.array([
    [0,0], #Sun
    [0, 0.0566], #Mercury
    [0,0.035], #Venus
    [0,0.03],#Earth
    [0,0.0265],#Mars
    [0,0.01372],#Jupiter
    [0,0.01014],#Saturne
    [0,0.00713],#Uranus
    [0,0.00547 ] #Neptune
    ], dtype = np.float64)

m = np.array([
    199*10**10, #Sun
    0.33*10**6, #Mercury
    4.87*10**6, #Venus
    5.97*10**6,#Earth
    0.642*10**6,#Mars
    1898*10**6,#Jupiter
    568*10**6,#Saturne
    86.8*10**6,#Uranus
    102*10**6 #Neptune
], dtype = np.float64)

rx = r[:,0]
ry = r[:,1]
vx = v[:,0]
vy = v[:,1]

t = np.arange(0, 5*31.536*10**6, 86400/5)
t = t.astype(np.float64)

rx = rx.astype(np.float64)
ry = ry.astype(np.float64)
m = m.astype(np.float64)

#call synchronize to initialize GPU for the first time
#and not to include this initialization time in the time measured
cuda.Context.synchronize()
tstart = time.time()

mod = SourceModule("""

    __global__ void calc_accelerationGPU(double* rx, double* ry, double* m, double* g_idatax, double* g_idatay,
                                                               double* ax_res, double* ay_res, unsigned int n){

        double const G = 6.6743 * pow(10.0, -11.0);
        unsigned int tid = threadIdx.x;
        unsigned int j = tid;
        unsigned int i = blockIdx.x;
        unsigned int idx = tid + blockIdx.x*blockDim.x;
        if (j >= n) return;
        if (i >= n) return;
        if (i == j) {
          g_idatax[idx] = 0;
          g_idatay[idx] = 0;
        }
        else {
          g_idatax[idx] = m[j]*(rx[j] - rx[i])/pow(sqrt( (rx[j] - rx[i])*(rx[j] - rx[i]) + (ry[j] - ry[i])*(ry[j] - ry[i]) ), 3.0);
          g_idatay[idx] = m[j]*(ry[j] - ry[i])/pow(sqrt( (rx[j] - rx[i])*(rx[j] - rx[i]) + (ry[j] - ry[i])*(ry[j] - ry[i]) ), 3.0);
        }
        double *idatax = g_idatax + blockIdx.x*blockDim.x;
        double *idatay = g_idatay + blockIdx.x*blockDim.x;
        for (int s = blockDim.x/2; s>0; s>>=1){
            if(tid<s){
                idatax[tid] += idatax[tid+s];
                idatay[tid] += idatay[tid+s];
            }
            __syncthreads();
        }
        if (tid == 0) {
          ax_res[i] = idatax[0] * G;
          ay_res[i] = idatay[0] * G;
        }
    }
    """)

positions_x, positions_y, velocities_x, velocities_y = solve_odeGPU(N, rx, ry, vx, vy, m, t)

tend = time.time()
print(tend - tstart)
# print("acceleration:")
# print(positions_x)
# print(positions_y)
#print(positions_x[2,:])
thin_t = thinout_array(t, 0, t.shape[0] - 1, step = 25)
thin_positions_x = thinout_array(positions_x, 0, t.shape[0] - 1, step = 25)
thin_positions_y = thinout_array(positions_y, 0, t.shape[0] - 1, step = 25)
thin_velocities_x = thinout_array(velocities_x, 0, t.shape[0] - 1, step = 25)
thin_velocities_y = thinout_array(velocities_y, 0, t.shape[0] - 1, step = 25)
np.save('./sol_GPUverle_px.npy', thin_positions_x)
np.save('./sol_GPUverle_py.npy', thin_positions_y)
np.save('./sol_GPUverle_vx.npy', thin_velocities_x)
np.save('./sol_GPUverle_vy.npy', thin_velocities_y)