import numpy as np
from PIL import Image
from helper_functions import *

def local_convolution(A, K, i, j):
    v=0
    for m in range(5):
        for n in range(5):
            v += A[i -2 + m, j -2 + n] * K[m, n]
    return v

def im_filtering(A, K):
    rows, cols = A.shape
    # Create an output matrix for the result
    R = np.zeros(shape=(rows - 4, cols - 4))

    for i in range(2, rows - 2):      # internal rows
        for j in range(2, cols - 2):  # internal columns
            R[i - 2, j - 2] = local_convolution(A, K, i, j)
    return R

# Load the image and convert it to gray scale
A = np.array(Image.open("plate.png").convert('L'))

# Create the convolution kernel
K = np.ones((5, 5)) / 25

# Padding
Ap = np.zeros((A.shape[0] + 4, A.shape[1] + 4))
Ap[2:-2, 2:-2] = A

R = im_filtering(Ap, K)
compare_images(A, R)