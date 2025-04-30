import matplotlib.pyplot as plt   # import package for plot
from math import sin              # import for sine function

# Physical parameters
A_t = 1      # tank cross-sectional area (m^2)
A_out = 0.1  # outlet area (m^2)
g = 9.81     # gravity
h0 = 1      # initial water level

# Simulation time parameters
T = 100       # total simulation time 
N = 1000      # number of time steps
dt = T / N    # time step

# outflow rate Q_out
Q_out = lambda h : A_out * (2 * g * max(0, h)) ** 0.5

# inflow rate
Q_in1 = lambda t: 1+sin(t/4)
Q_in2 = lambda t: 1+sin(t/8)
Q_in3 = lambda t: 1.5*(t<=50) + 0.5*(t>50)
labels = ['1+sin(t/4)', '1+sin(t/8)', 'Piecewise constant']  # labels for the legend

plt.figure(figsize=(10, 5))
# loop over different inflow functions
for j, Q_in in enumerate([Q_in1, Q_in2, Q_in3]):
    # lists to store the solution 
    times = (N+1)*[0]
    h = (N+1)*[0]
    h[0] = h0
    
    for i in range(N):
        dh = dt * (Q_in(i*dt) - Q_out(h[i]))
        h[i+1] = h[i] + dh 
        times[i+1] = (i+1)*dt
    
    plt.plot(times, h, label = labels[j])

plt.xlabel("Time (s)")
plt.ylabel("Water Level (m)")
plt.title("Water Level Over Time for Different Q_in")
plt.legend()
plt.grid()
plt.show()
    