function e_bar = E_bar(y)
    y_bar = sum(y) ./ length(y);  % Mean of the dataset
    e_bar = sum((y - y_bar) .^ 2);
end

function error = E(x, y)
    [m, q] = my_linear_regression(x, y);
    y_hat = m.*x + q;  % Predicted values
    error = sum((y - y_hat) .^ 2);
end

function r_2 = R2(x, y)
    e_bar = E_bar(y);
    error = E(x, y);
    r_2 = 1 - error / e_bar;
end

my_dataset = readtable('Algerian_forest_dataset.csv');
Temperature = my_dataset.Temperature;
RH = my_dataset.RH;
BUI = my_dataset.BUI;
FWI = my_dataset.FWI;

r_2_Temperature = R2(Temperature, FWI);
r_2_RH = R2(RH, FWI);
r_2_BUI = R2(BUI, FWI);

fprintf('Coefficient of determination for Temperature-FWI regression: %f\n', r_2_Temperature);
fprintf('Coefficient of determination for RH-FWI regression: %f\n', r_2_RH);
fprintf('Coefficient of determination for BUI-FWI regression: %f\n', r_2_BUI);
