import numpy as np
import pandas as pd

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

def E_bar(y):
    y_bar = np.sum(y) / len(y)  # Mean of the dataset
    e_bar = np.sum((y - y_bar) ** 2)
    return e_bar

def E(x, y):
    m, q = linear_regression(x, y)
    y_hat = m*x + q  # Predicted values
    error = np.sum((y - y_hat)**2)
    return  error

def R2(x, y):
    e_bar = E_bar(y)
    error = E(x, y)
    r_2 = 1 - error / e_bar
    return r_2

my_dataset = pd.read_csv('Algerian_forest_dataset.csv')
Temperature = my_dataset['Temperature'].values
RH = my_dataset['RH'].values
BUI = my_dataset['BUI'].values
FWI = my_dataset['FWI'].values

r_2_Temperature = R2(Temperature, FWI)
r_2_RH = R2(RH, FWI)
r_2_BUI = R2(BUI, FWI)

print(f"Coefficient of determination for Temperature-FWI regression: {r_2_Temperature}")
print(f"Coefficient of determination for RH-FWI regression: {r_2_RH}")
print(f"Coefficient of determination for BUI-FWI regression: {r_2_BUI}")