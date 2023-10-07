import numpy as np
import matplotlib.pyplot as plt
import math

def fu(x):
  return 1 / (1 + 25 * x * x)

def polynomial_coefficients(xs, coeffs):
  ys = 1
  for i in range(len(coeffs)):
    ys = ys * (xs - coeffs[i])
  return ys

def poly_lagranj(x1, x):
  z = 0
  for j in range(jj):
    p1 = 1; p2 = 1
    for i in range(jj):
      if i != j:
        p1 = p1 * (x - x1[i])
        p2 = p2 * (x1[j] - x1[i])
    z = z + fu(x1[j]) * p1 / p2
  return z

def poly_newton(x1, x):
  h = x1[1] - x1[0]
  y = [fu(ii) for ii in x1]
  z = y[0]
  dx = 1
  for j in range(1, jj+1):
    dx *= x - x1[j-1]
    for i in range(len(y)-1):
      y[i] = y[i+1] - y[i]
    z += dx * y[0] / (math.factorial(j) * h ** j)
  return z
  
xs = np.linspace(1, 6, 500)
n = [3, 5, 10, 20]
for j in n:
  coeffs = np.linspace(1, 6, j)
  plt.style.use("fivethirtyeight")
  plt.plot(xs, polynomial_coefficients(xs, coeffs))
  plt.xlabel(r'$x$', fontsize=14)
  plt.ylabel(r'$w(x)$', fontsize=14)
  plt.grid(True)
  plt.show()

x = np.linspace(-1, 1, 500)

k = [5, 8, 15, 16]

for jj in k:
  x1 = np.linspace(-1, 1, jj)
  plt.plot(x, np.array([fu(ii) for ii in x]), color='r', label='Функция')
  plt.plot(x, np.array([poly_lagranj(x1, ii) for ii in x]), color='g', label='Интерполяция Лагранжа')
  plt.plot(x, np.array([poly_newton(x1, ii) for ii in x]), color='b', label='Интерполяция Ньютона')
  plt.xlabel(r'$x$', fontsize=14)
  plt.ylabel(r'$f(x)$', fontsize=14)
  plt.grid(True)
  plt.legend(loc='best', fontsize=12)
  plt.show()
