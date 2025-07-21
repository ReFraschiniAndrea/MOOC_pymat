function ss_tot = SStot(y)
    y_bar = sum(y) ./ length(y);  % Mean of the dataset
    ss_tot = sum((y - y_bar) .^ 2);
end

function error = E(x, y)
    [m, q] = my_linear_regression(x, y);
    y_hat = m.*x + q;  % Predicted values
    error = sum((y - y_hat) .^ 2);
end

function r_2 = Rsquared(x, y)
    ss_tot = SStot(y);
    error = E(x, y);
    r_2 = 1 - error / ss_tot;
end

my_dataset = readtable('Algerian_forest_dataset.csv');
Temperature = my_dataset.Temperature;
RH = my_dataset.RH;
FWI = my_dataset.FWI;

r_2_Temperature = Rsquared(Temperature, FWI);
r_2_RH = Rsquared(RH, FWI);

fprintf('Coefficient of determination for Temperature-FWI regression: %f\n', r_2_Temperature);
fprintf('Coefficient of determination for RH-FWI regression: %f\n', r_2_RH);
