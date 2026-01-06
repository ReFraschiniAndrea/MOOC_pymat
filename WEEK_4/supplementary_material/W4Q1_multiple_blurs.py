import numpy as np
from PIL import Image
from helper_functions import *

def local_convolution(A, K, i, j):
    v=0
    for m in range(3):
        for n in range(3):
            v += A[i -1 + m, j -1 + n] * K[m, n]
    return v

def im_filtering(A, K):
    rows, cols = A.shape
    # Create an output matrix for the result
    R = np.zeros(shape=(rows - 2, cols - 2))

    for i in range(1, rows - 1):      # internal rows
        for j in range(1, cols - 1):  # internal columns
            R[i - 1, j - 1] = local_convolution(A, K, i, j)
    return R

# Load the image and convert it to gray scale
A = np.array(Image.open("plate.png").convert('L'))

# Create the convolution kernel
K = np.ones((3, 3)) / 9

# Padding
Ap = np.zeros((A.shape[0] + 2, A.shape[1] + 2))
Ap[1:-1, 1:-1] = A

# Number of times to apply the blur
N = 20

for i in range(N):
    R = im_filtering(Ap, K)
    # Reapply the padding to the blurred image
    Ap[1:-1, 1:-1] = R

compare_images(A, R)