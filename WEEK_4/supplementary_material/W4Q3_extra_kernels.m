% Load the image and convert it to gray scale
A = double(rgb2gray(imread("plate.png")));

% Create the convolution kernels
K_edge_detection = [
    -1 2 -1
    -1 2 -1
    -1 2 -1
];
K_sharpening = [
     0, -1,  0;
    -1,  5, -1;
     0, -1,  0;
];

% Padding
Ap = zeros(size(A, 1) + 2, size(A, 2) + 2);
Ap(2 : end - 1, 2 : end - 1) = A;

R_edge_detection = im_filtering(Ap, K_edge_detection);
R_sharpening = im_filtering(Ap, K_sharpening);
compare_images(A, uint8(R_edge_detection))
compare_images(A, uint8(R_sharpening))