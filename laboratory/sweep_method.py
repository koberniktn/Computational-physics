import numpy as np
import matplotlib.pyplot as plt

def answer(x):
  return np.exp(- 2 * x) + np.exp(x)

def runge(x):
  return 1/(1 + x ** 2)

def runge_diff(x):
  return (-2 * x) / ((1 + x ** 2) ** 2)

def num_1():
  x = np.linspace(0, 1, 501)
  h = x[1] - x[0]
  alpha = 1 + h / 2
  beta = - 2 * (h ** 2) - 2
  gamma = 1 - h / 2
  A, B = [0], [2]
  y = [1 for j in range(len(x))]
  y[0] = 2
  y[-1] = (np.exp(-2) + np.exp(1))

  for i in range(1, len(x) - 1):
    B.append((-gamma * B[-1]) / (beta + gamma * A[-1]))
    A.append(- alpha / (beta + gamma * A[-1]))

  for k in range(1, len(x) - 1):
    y[-k-1] = y[-k]*A[-k] + B[-k]

  plt.plot(x, y, color='r', label='Метод прогонки')
  plt.plot(x, answer(x), color='g', label='Аналитическое решение')
  plt.xlabel(r'$x$', fontsize=14)
  plt.ylabel(r'$f(x)$', fontsize=14)
  plt.grid(True)
  plt.legend(loc='best', fontsize=12)
  plt.show()

def Spline_sec(x, x1, m, kk):
  h = x1[1] - x1[0]
  m_i = m[20*kk]
  m_i_next = m[20*kk + 20]
  S3 = ((x1[1] - x) ** 2) * (2 * (x - x1[0]) + h) * runge(x1[0]) / (h ** 3) \
  + (((x - x1[0]) ** 2) * (2 * (x1[1] - x) + h) * runge(x1[1]) / (h ** 3)) \
  + (((x1[1] - x) ** 2) * (x - x1[0]) * m_i / (h ** 2)) \
  + (((x - x1[0]) ** 2) * (x - x1[1]) * m_i_next / (h ** 2))
  return S3

def num_2():
  x = np.linspace(-4, 4, 401)
  x1 = np.linspace(-4, 4, 21)
  h = x[1] - x[0]
  m = [1 for j in range(len(x))]
  m[0] = (-3 * runge(x[0]) + 4 * runge(x[1]) - runge(x[2])) / (2 * h)
  m[-1] = (3 * runge(x[-1]) - 4 * runge(x[-2]) + runge(x[-3])) / (2 * h)
  alpha = 1
  beta = 4
  gamma = 1
  # C1 = 1, d1 = 1, C2 = 0, d2 = 0
  A, B = [0], [runge(x[0])]

  for i in range(1, len(x) - 1):
    delta = (3 / h) * (runge(x[i + 1]) - runge(x[i - 1]))
    B.append((delta - gamma * B[-1]) / (beta + gamma * A[-1]))
    A.append(- alpha / (beta + gamma * A[-1]))

  for k in range(1, len(x) - 1):
    m[-k-1] = m[-k]*A[-k] + B[-k]

  for kk in range(x1.size-1):
    x = np.linspace(x1[kk], x1[kk + 1], round(500/x1.size))

  plt.plot(x, np.array([runge(ii) for ii in x]), color='r')
  plt.plot(x, np.array([Spline_sec(ii, [x1[kk], x1[kk + 1]], m, kk) for ii in x]), color='b')
  plt.xlabel(r'$x$', fontsize=14)
  plt.ylabel(r'$f(x)$', fontsize=14)
  plt.grid(True)
  plt.legend(loc='best', fontsize=12)
  plt.show()

num_1()
num_2()
