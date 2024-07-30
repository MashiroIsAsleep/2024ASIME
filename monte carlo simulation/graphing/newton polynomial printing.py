
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
from sympy import symbols, Eq
from sympy.printing.latex import latex
from numpy.polynomial import Polynomial

def newton_interpolation(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i,j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])

    return coef[0, :]

def newton_polynomial(coef, x_data):
    n = len(coef)
    equation = f"{coef[0]:.3f}"

    for i in range(1, n):
        term = f"{coef[i]:+.3f}"
        for j in range(i):
            term += f"*(x-{x_data[j]:.3f})"
        equation += term

    return equation

# Replace this with your actual data
x = average.index
y = average
coef = newton_interpolation(x, y)
equation = newton_polynomial(coef, x)
print(equation)