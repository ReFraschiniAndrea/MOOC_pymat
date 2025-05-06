% Function to define the inflow rate function
Q_in = @(t) 1;
% Function to calculate the outflow rate based on the water level and outlet area
Q_out = @(outlet_area, water_level) sqrt(2*9.81 * max(water_level,0)) * outlet_area;
A_out = 0.1; % Outlet area (m2)
A_t = 1; % Tank cross-sectional area (m2)
h0 = 1; % Initial water level

% Simulation time parameters
T = 100; % Total simulation time (seconds)

% Create figure for plotting
figure()
grid on
hold on

% Loop over different time step sizes
for dt = [0.01, 0.1, 1, 5, 10, 20]
  N = round(T / dt); % Number of time steps
  % Create arrays to store data for plotting
  times = zeros(1, N + 1);
  h = zeros(1, N + 1);
  h(1) = h0;
  t = 0;
  % Simulate the water level over time
  for i = 1:N
      t = t + dt;
      h(i+1) = h(i) + (Q_in(t) - Q_out(A_out, h(i))) * dt;
      times(i+1) = t;
  end
  plot(times, h, '-')
end

legend('dt=0.01', 'dt=0.1', 'dt=1', 'dt=5', 'dt=10', 'dt=20')

