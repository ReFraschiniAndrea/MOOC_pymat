import matplotlib.pyplot as plt   # import package for plot

# Physical parameters
Q_in = 1     #inflow rate
A_t = 1      # tank cross-sectional area (m^2)
A_out = 0.1  # outlet area (m^2)
g = 9.81     # gravity
h0 = 1      # initial water level

# Simulation time parameters
T = 100       # total simulation time 

# outflow rate Q_out
def Q_out(h):
    return A_out * (2 * g * h) ** 0.5

plt.figure(figsize=(10, 5))

# loop over different values for the time step
for dt in [0.01, 0.1, 1, 5, 10, 20]:
    N = int(T / dt)   # number of time steps

    # lists to store the solution 
    times = (N+1)*[0]
    h = (N+1)*[0]
    h[0] = h0
    
    for i in range(N):
        dh = dt * (Q_in - Q_out(h[i]))
        h[i+1] = h[i] + dh 
        times[i+1] = (i+1)*dt
    
    plt.plot(times, h, label = 'dt = ' + str(dt))

plt.xlabel("Time (s)")
plt.ylabel("Water Level (m)")
plt.title("Water Level Over Time for Different dt")
plt.legend()
plt.grid()
plt.show()
    