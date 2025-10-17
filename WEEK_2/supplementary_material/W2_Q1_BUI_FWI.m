my_dataset = readtable('Algerian_forest_dataset.csv');
x = my_dataset.BUI;
y = my_dataset.FWI;

[m, q] = my_linear_regression(x, y);

% Print results
disp('Linear model results:');
fprintf('Slope (m): %f\n', m);
fprintf('Y-intercept (q): %f\n', q);

figure;
scatter(x, y, 'blue', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(x, m*x+q, 'red', 'LineWidth', 2);
xlabel('Temperature');
ylabel('FWI');
title('Linear Regression: FWI vs BUI');
grid on;
legend('Data points', 'Regression line');