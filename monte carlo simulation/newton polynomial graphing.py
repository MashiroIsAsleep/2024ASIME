import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, Eq
from sympy.printing.latex import latex
from numpy.polynomial import Polynomial

def _poly_newton_coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a

def newton_polynomial(x_data, y_data, x):
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p
    return p

# Replace this with your actual data
x_data = average.index
y_data = average


x_range = np.linspace(min(x_data), max(x_data), 100)
y_range = [newton_polynomial(x_data, y_data, x) for x in x_range]


plt.plot(x_range, y_range, label='Newton Polynomial')
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton Polynomial Interpolation')
plt.legend()
plt.grid(True)
plt.show()
