import numpy as np
import matplotlib.pyplot as plt
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

my_dataset = pd.read_csv('Algerian_forest_dataset.csv')
x = my_dataset['BUI'].values
y = my_dataset['FWI'].values
m, q = linear_regression(x, y)

print('Linear model results:')
print(f'Slope (m): {m}')
print(f'Y-intercept (q): {q}')

plt.figure(figsize=(16, 10))
plt.scatter(x, y, color='blue', alpha=0.5, label='Data points')
plt.plot(x, m*x+q, color='red', label='Regression line')

plt.xlabel('Temperature')
plt.ylabel('FWI')
plt.title('Linear Regression: FWI vs BUI')
plt.grid(True)
plt.legend()
plt.show()