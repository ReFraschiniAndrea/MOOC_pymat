% Load the dataset
my_dataset = readtable('Algerian_forest_dataset.csv');

% Dataset dimensions
disp('Shape of the dataset:');
disp(size(my_dataset));

% Showing the data
disp('First 5 rows of the dataset:');
disp(my_dataset(1:5, :));

% Perform simple linear regression
x = my_dataset.Temperature;
y = my_dataset.FWI;

[m, q] = my_linear_regression(x, y);

% Print results
disp('Linear model results:');
fprintf('Slope (m): %f\n', m);
fprintf('Y-intercept (q): %f\n', q);

% Plotting
figure;
scatter(x, y, 'blue', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(x, m*x+q, 'red', 'LineWidth', 2);

xlabel('Temperature');
ylabel('FWI');
title('Linear Regression: FWI vs Temperature');
grid on;
legend('Data points', 'Regression line');

% Perform simple linear regression
x = my_dataset.RH;
y = my_dataset.FWI;

[m, q] = my_linear_regression(x, y);

% Print results
disp('Linear model results:');
fprintf('Slope (m): %f\n', m);
fprintf('Y-intercept (q): %f\n', q);

% Plotting
figure;
scatter(x, y, 'blue', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(x, m*x+q, 'red', 'LineWidth', 2);

xlabel('Temperature');
ylabel('FWI');
title('Linear Regression: FWI vs RH');
grid on;
legend('Data points', 'Regression line');