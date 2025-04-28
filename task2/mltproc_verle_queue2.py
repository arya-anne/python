import multiprocessing
from multiprocessing import Process
from multiprocessing import RawArray
import os
import numpy as np
import time
from multiprocessing import Queue
import queue

#print("Number of cpu : ", multiprocessing.cpu_count())

name='bob'

rmin = 0
rmax = 300
vmin = -5
vmax = 5
mmin = 1
mmax = 5000000
#N = 100

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

def rand_initial_values(N, rmin, rmax, vmin, vmax, mmin, mmax):
  r = rmin + (rmax - rmin)*np.random.rand(N, 2)
  v = vmin + (vmax - vmin)*np.random.rand(N, 2)
  m = mmin + (mmax - mmin)*np.random.rand(N, 1)
  return r, v, m

def f():
    print(f'pid1= %d', os.getpid())
    print('hello', name)

def calc_acceleration(tasks, tasks_done):
  G = 6.6743 * 10**(-11)
  #print('hello')
#   print(f'slave pid = {os.getpid()}, N1 = {N1}, N2 = {N2}')
  while True:
      task = tasks.get()
      if (task == 'STOP'):
         break
      rx, ry, m, N1, N2 = task
      #print(f'slave pid = {os.getpid()}, N1 = {N1}, N2 = {N2}')
      ax = np.zeros(N2 - N1)
      ay = np.zeros(N2 - N1)
      for n in range(N1, N2):
         for k in range(N):
            if n == k:
               continue
            if rx[n] == ry[k] and ry[n]==ry[k]:
               continue
            #print(f'slave pid = {os.getpid()}, N1 = {N1}, N2 = {N2}, n={n}')
            ax[n - N1] = ax[n - N1] + m[k].item()*(rx[k] - rx[n])/(((rx[k] - rx[n])**2 + (ry[k] - ry[n])**2)**0.5)**3
            ay[n - N1] = ay[n - N1] + m[k].item()*(ry[k] - ry[n])/(((rx[k] - rx[n])**2 + (ry[k] - ry[n])**2)**0.5)**3
            #print(f'slave pid = {os.getpid()}, N1 = {N1}, N2 = {N2}, n={n}, ax[n - N1] = {ax[n - N1]}, ay[n - N1] = {ay[n- N1]}')
      ax = ax*G
      ay = ay*G
      tasks_done.put((ax, ay, N1, N2))
      #print(f'finished slave pid = {os.getpid()}, N1 = {N1}, N2 = {N2}')
  #return ax, ay

def init_procs_and_queue(Nproc):
   processes = []
   tasks = Queue()
   tasks_done = Queue()
   chunk_size = int (np.floor(N/Nproc))
   Nchunks = Nproc
   last_chunk_size = N - (Nchunks-1)*chunk_size
   for i in range(Nproc):
      p = Process(target=calc_acceleration, args=(tasks, tasks_done))
      processes.append(p)
      p.start()
   return tasks, tasks_done, chunk_size, last_chunk_size

def stop_procs(Nproc, tasks):
   for i in range(Nproc):
      tasks.put('STOP')

def calc_acceleration_mltproc(rx, ry, m, Nproc, tasks, tasks_done, chunk_size, last_chunk_size):
   if __name__ == '__main__':
      for i in range(0, Nproc):
         if i == Nproc - 1:
            tasks.put((rx, ry, m, N - last_chunk_size, N))
         else:
            tasks.put((rx, ry, m, i*chunk_size, i*chunk_size + chunk_size))
      # for i in range(Nproc):
      #    p.join()
      ax = np.zeros(N)
      ay = np.zeros(N)
      for i in range(Nproc):
         ax0, ay0, N1, N2 = tasks_done.get()
         # print(ax0)
         # print(ay0)
         ax[N1:N2] = ax0
         ay[N1:N2] = ay0
      #print(ax)
      return ax, ay
   
def Verlet(rx, ry, vx, vy, ax, ay, dt, m, Nproc, tasks, tasks_done, chunk_size, last_chunk_size ):
  rx_n1 = rx + vx*dt + ax/2*dt**2
  ry_n1 = ry + vy*dt + ay/2*dt**2

  ax_n1, ay_n1 = calc_acceleration_mltproc(rx_n1, ry_n1, m, Nproc, tasks, tasks_done, chunk_size, last_chunk_size)
  vx_n1 = vx + 1/2*(ax+ax_n1)*dt
  vy_n1 = vy + 1/2*(ay+ay_n1)*dt
  return rx_n1, ry_n1, vx_n1, vy_n1, ax_n1, ay_n1

def solve_ode(N, rx0, ry0, vx0, vy0, m, t, Nproc):
  rx = np.zeros((N, t.shape[0]))
  ry = np.zeros((N, t.shape[0]))
  vx = np.zeros((N, t.shape[0]))
  vy = np.zeros((N, t.shape[0]))

  tasks, tasks_done, chunk_size, last_chunk_size = init_procs_and_queue(Nproc)

  ax, ay = calc_acceleration_mltproc(rx0, ry0, m, Nproc, tasks, tasks_done, chunk_size, last_chunk_size)
  rx[:,0] = rx0
  ry[:,0] = ry0
  vx[:,0] = vx0
  vy[:,0] = vy0
  for i in range(1, t.shape[0]):
    rx[:,i], ry[:,i], vx[:,i], vy[:,i], ax, ay = Verlet(rx[:,i-1], ry[:,i-1],
                                                         vx[:,i-1], vy[:,i-1],
                                                         ax, ay, t[i] - t[i-1], m, 
                                                         Nproc, tasks, tasks_done, chunk_size, last_chunk_size)
  stop_procs(Nproc, tasks)
  return rx, ry, vx, vy

if __name__ == '__main__':
   #print(f'pid0= %d', os.getpid())
    #r, v, m = rand_initial_values(N, rmin, rmax, vmin, vmax, mmin, mmax)
    # time.sleep(2)
   #  r = np.array([[50,50], [50, 40], [50, 30], [50, 20]])
   #  v = np.array([[0, 0], [3, 0], [4, 0], [5, 0]])
   #  m = np.array([1.35*10**12, 10, 10, 10])
   N = 400
   r = np.load('./r_400.npy')
   v = np.load('./v_400.npy')
   m = np.load('./m_400.npy')

   # #Solar system
   # cx = 5000*10**3
   # cy = 5000*10**3
   # a = 4514.953*10**3
   # e = 0.0097
   # b = a * (1 - e**2)**0.5
   # Sx = cx + a*e
   # Sy = cy
   # r = np.array([
   #    [Sx, Sy], #Sun
   #    [Sx + 45.9*10**3, Sy], #Mercury
   #    [Sx + 107.476*10**3, Sy], #Venus
   #    [Sx + 47.098*10**3, Sy],#Earth
   #    [Sx + 206.665*10**3, Sy],#Mars
   #    [Sx + 740.595*10**3, Sy],#Jupiter
   #    [Sx + 1357.554*10**3, Sy],#Saturne
   #    [Sx + 2732.696*10**3, Sy],#Uranus
   #    [Sx + 4471.050*10**3, Sy] #Neptune
   #    ], dtype = np.float64)
   # v = np.array([
   #    [0,0], #Sun
   #    [0, 0.0566], #Mercury
   #    [0,0.035], #Venus
   #    [0,0.03],#Earth
   #    [0,0.0265],#Mars
   #    [0,0.01372],#Jupiter
   #    [0,0.01014],#Saturne
   #    [0,0.00713],#Uranus
   #    [0,0.00547 ] #Neptune
   #    ], dtype = np.float64)

   # m = np.array([
   #    199*10**10, #Sun
   #    0.33*10**6, #Mercury
   #    4.87*10**6, #Venus
   #    5.97*10**6,#Earth
   #    0.642*10**6,#Mars
   #    1898*10**6,#Jupiter
   #    568*10**6,#Saturne
   #    86.8*10**6,#Uranus
   #    102*10**6 #Neptune
   # ], dtype = np.float64)

   rx = r[:,0]
   ry = r[:,1]
   vx = v[:,0]
   vy = v[:,1]

   t = np.arange(0, 100, 1)
   #t = np.arange(0, 5*31.536*10**6, 86400/5)
   tstart = time.time()
   positions_x, positions_y, velocities_x, velocities_y = solve_ode(N, rx, ry, vx, vy, m, t, Nproc = 6)
   tend = time.time()
   print(tend - tstart)
   #print(positions_x[1:])
   #  print(ax)
   #  print(ay)
    # p = Process(target=f, args=())
    # p.start()
    # p.join()
   # thin_t = thinout_array(t, 0, t.shape[0] - 1, step = 25)
   # thin_positions_x = thinout_array(positions_x, 0, t.shape[0] - 1, step = 25)
   # thin_positions_y = thinout_array(positions_y, 0, t.shape[0] - 1, step = 25)
   # thin_velocities_x = thinout_array(velocities_x, 0, t.shape[0] - 1, step = 25)
   # thin_velocities_y = thinout_array(velocities_y, 0, t.shape[0] - 1, step = 25)
   # np.save('./sol_mulverle_px.npy', thin_positions_x)
   # np.save('./sol_mulverle_py.npy', thin_positions_y)
   # np.save('./sol_mulverle_vx.npy', thin_velocities_x)
   # np.save('./sol_mulverle_vy.npy', thin_velocities_y)