% Function to define the inflow rate function
Q_in = @(t) 1;
% Function to calculate the outflow rate based on the water level and outlet area
Q_out = @(A_out, h) sqrt(2*9.81 * max(h,0)) * A_out;
A_out = 0.1; % Outlet area (m2)
A_t = 1; % Tank cross-sectional area (m2)

% Simulation time parameters
T = 100; % Total simulation time (seconds)
N = 1000; % Number of time steps
% Calculate the time step size
dt = T / N;

% Create figure for plotting
figure()
grid on
hold on

% Loop over different initial water levels
for h0 = [0, 1, 2, 5, 8]
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

legend('h0=0', 'h0=1', 'h0=2', 'h0=5', 'h0=8')

