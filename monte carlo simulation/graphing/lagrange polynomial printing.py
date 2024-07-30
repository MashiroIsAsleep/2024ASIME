import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
from sympy import symbols, Eq
from sympy.printing.latex import latex
from numpy.polynomial import Polynomial

# Replace this with your actual data
X = average.index.to_numpy()
Y = average.to_numpy()

n = len(X)
poly = Polynomial(np.zeros(n))

for j in range(n):
    k = [k for k in range(n) if k != j]
    roots = -1 * X[k]

    sub_poly = Polynomial.fromroots(X[k])
    scale = Y[j] / np.prod(X[j] - X[k])
    sub_poly.coef *= scale

    poly.coef += sub_poly.coef
    
plt.scatter(X, Y)
Xinterp = np.linspace(min(X), max(X), 100)
plt.plot(Xinterp, poly(Xinterp))
plt.show()
