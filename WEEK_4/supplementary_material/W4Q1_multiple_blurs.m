% Load the image and convert it to gray scale
A = double(rgb2gray(imread("plate.png")));

% Create the convolution kernel
K = ones(3, 3) / 9;

% Padding
Ap = zeros(size(A, 1) + 2, size(A, 2) + 2);
Ap(2 : end - 1, 2 : end - 1) = A;

% Number of times to apply the blur
N = 20;

for i = 1 : 20
    R = im_filtering(Ap, K);
    % Reapply the padding to the blurred image
    Ap(2 : end - 1, 2 : end - 1) = R;
end

R = uint8(R);
compare_images(A, R)