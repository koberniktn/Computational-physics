import math
import numpy as np
import matplotlib.pyplot as plt

def method_of_edges(x0, k):
  xn = x0 - f(x0)/f1(x0)
  for i in range(k):
    x0 = xn
    xn = x0 - f(x0)/f1(x0)
  return x0

def easy_method_of_edges(x0, k):
  x0_const = x0
  xn = x0 - f(x0)/f1(x0_const)
  for i in range(k):
  x0 = xn
  xn = x0 - f(x0)/f1(x0_const)
  return x0
  
a = 0.3
b = 0.8
eps = 0.0001

print('Введите количество итераций')
k = int(input())

f = lambda x: x * math.cos(3 * x) # y = f(x)
f1 = lambda x: (f(x + eps) - f(x - eps)) / (2 * eps) # y' = f'(x)
f2 = lambda x: (f1(x + eps) - f1(x - eps)) / (2 * eps) # y'' = f''(x)

if f(a) / f2(a) > 0:
  x0 = a
else:
  x0 = b
  
y=lambda x: x * np.cos(3 * x)
y1=lambda x: np.cos(3 * x) - 3 * x * np.sin(3 * x)
y2=lambda x: -6 * np.sin(3 * x) - 9 * x * np.cos(3 * x)

fig = plt.subplots()
x = np.linspace(0.2, 0.9,10000)
plt.plot(x, y(x), label=r'$f(x)=x*cos(3x)$')
plt.plot(x,y1(x), label=r'$df/dx$')
plt.plot(x,y2(x), label=r'$d^2f/dx^2$')
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$f(x)$', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.show()
x0 = easy_method_of_edges(x0, k)
print('По упрощенному методу Ньютона:\nx0 =', x0)
print('f(x0) =', f(x0))
x0 = method_of_edges(x0, k)
print('По методу касательных:\nx0 =', x0)
print('f(x0) =', f(x0))
