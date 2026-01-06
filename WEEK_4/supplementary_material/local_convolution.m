function [v] = local_convolution(A, K, i, j)
    v = 0;
    for m = 1 : 3      % kernel rows
        for n = 1 : 3  % kernel columns
            v = v + A(i - 2 + m, j - 2 + n) * K(m, n);
        end
    end
end