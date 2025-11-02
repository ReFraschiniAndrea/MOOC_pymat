% Data
Q_in = 1;     % inflow rate
A_t = 1;      % tank cross-sectional area (m^2)
A_out = 0.1;  % outlet area (m^2)
g = 9.81;     % gravity
h_0 = 0;       % initial water level

% Simulation time parameters
T = 100;      % total simulation time 
N = 1000;     % number of time steps
dt = T / N;

% Outflow rate
Q_out = @(h) A_out * (2 * g * h) ^ 0.5;

% Test
disp('Outflow rate with h=0.5 is')
disp(Q_out(0.5))

% Arrays to store the solution
times = zeros(1, N + 1);
h = zeros(1, N + 1);
h(1) = h_0;

% Loop
for n = 1 : N
    dh = dt * (Q_in - Q_out(h(n))) / A_t;
    h(n+1) = h(n) + dh;
    times(n+1) = n * dt;
end

% Plot
plot(times, h, 'b')
grid on
xlabel('t')
ylabel('h')