% Function to calculate the outflow rate based on the water level and outlet area
Q_out = @(A_out, h) sqrt(2*9.81 * max(h,0)) * A_out;
A_out = 0.1; % Outlet area (m2)
A_t = 1; % Tank cross-sectional area (m2)

% Simulation time parameters
T = 100; % Total simulation time (seconds)
N = 1000; % Number of time steps
% Calculate the time step size
dt = T / N;

% Functions to define the inflow rate function
Q_in1 = @(t) 1 + sin(t/4);
Q_in2 = @(t) 1 + sin(t/8);
Q_in3 = @(t) 1.5 * (t<=50) + 0.5 * (t > 50);
inflows = {Q_in1, Q_in2, Q_in3};

% Create figure for plotting
figure()
grid on
hold on

% Loop over different inflow functions
for i = 1:3
  Q_in = inflows{i};
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

legend('1+sin(t/4)', '1+sin(t/8)', 'Piecewise constant')

