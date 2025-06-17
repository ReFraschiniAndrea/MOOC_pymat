import numpy as np
def linear_regression(x, y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    n = len(x)
    numerator = n*sum_xy - sum_x*sum_y
    denominator = n*sum_x2 - sum_x**2
    m = numerator / denominator
    q = (sum_y - m*sum_x)/n
    
    return m, q