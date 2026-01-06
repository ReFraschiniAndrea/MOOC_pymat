function [R] = im_filtering(A, K)
    [rows, cols] = size(A);
    % Create an output matrix for the result
    R = zeros(rows - 2, cols - 2);

    for i = 2 : rows - 1      % internal rows
        for j = 2 : cols - 1  % internal columns
            R(i - 1, j - 1) = local_convolution(A, K, i, j);
        end
    end
end