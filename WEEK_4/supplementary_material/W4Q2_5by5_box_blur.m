function [v] = local_convolution(A, K, i, j)
    v = 0;
    for m = 1 : 5      % kernel rows
        for n = 1 : 5  % kernel columns
            v = v + A(i - 3 + m, j - 3 + n) * K(m, n);
        end
    end
end

function [R] = im_filtering(A, K)
    [rows, cols] = size(A);
    % Create an output matrix for the result
    R = zeros(rows - 4, cols - 4);

    for i = 3 : rows - 2     % internal rows
        for j = 3 : cols - 2  % internal columns
            R(i - 2, j - 2) = local_convolution(A, K, i, j);
        end
    end
end

% Load the image and convert it to gray scale
A = double(rgb2gray(imread("plate.png")));

% Create the convolution kernel
K = ones(5, 5) / 25;

% Padding
Ap = zeros(size(A, 1) + 4, size(A, 2) + 4);
Ap(3 : end - 2, 3 : end - 2) = A;

R = im_filtering(Ap, K);
R = uint8(R);
compare_images(A, R)
