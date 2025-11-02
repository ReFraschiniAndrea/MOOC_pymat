% Linear regression
function [m, q] = my_linear_regression(x, y)
    sum_x = sum(x);
    sum_y = sum(y);
    sum_xy = sum(x.*y);
    sum_x2 = sum(x.^2);
    n = length(x);
    numerator = n*sum_xy - sum_x*sum_y;
    denominator = n*sum_x2 - sum_x^2;
    m = numerator / denominator;
    q = (sum_y - m*sum_x)/n;
end