import matplotlib.pyplot as plt   # import package for plot

# Physical parameters
Q_in = 1     #inflow rate
A_t = 1      # tank cross-sectional area (m^2)
A_out = 0.1  # outlet area (m^2)
g = 9.81     # gravity

# Simulation time parameters
T = 100       # total simulation time 
N = 1000      # number of time steps
dt = T / N    # time step

# outflow rate Q_out
def Q_out(h):
    return A_out * (2 * g * h) ** 0.5

plt.figure(figsize=(10, 5))

# loopover different values of intial water level
for h0 in [0, 2, 5, 8, 10]:

    # lists to store the solution 
    times = (N+1)*[0]
    h = (N+1)*[0]
    h[0] = h0
    
    for i in range(N):
        dh = dt * (Q_in - Q_out(h[i]))
        h[i+1] = h[i] + dh 
        times[i+1] = (i+1)*dt
    
    plt.plot(times, h, label = 'h0 = '+str(h0))

plt.xlabel("Time (s)")
plt.ylabel("Water Level (m)")
plt.title("Water Level Over Time for Different h0")
plt.legend()
plt.grid()
plt.show()
    