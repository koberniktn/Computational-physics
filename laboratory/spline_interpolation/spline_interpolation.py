import numpy as np
import matplotlib.pyplot as plt
import math
 
def Runge(x):
  return 1 / (1 + 25 * x * x)
	 
def Runge_dif(x):
  return -50 * x / ((1 + 25 * x ** 2) ** 2)

def Spline_first(x, x1, flager):
  h = x1[1] - x1[0]
  if flager == 0:
    m_i = (-3 * Runge(x1[0]) + 4 * Runge(x1[1]) - Runge(x1[2])) / (2 * h)
    m_i_next = (Runge(x1[2]) - Runge(x1[0])) / (2 * h)
    i = 0
  elif flager == 2:
    m_i_next = (3 * Runge(x1[2]) - 4 * Runge(x1[1]) + Runge(x1[0])) / (2 * h)
    m_i = (Runge(x1[2]) - Runge(x1[0])) / (2 * h)
    i = 1
  else:
    m_i = (Runge(x1[2]) - Runge(x1[0])) / (2 * h)
    m_i_next = (Runge(x1[3]) - Runge(x1[1])) / (2 * h)
    i = 1
  
  S3 = (
  ((x1[i + 1] - x) ** 2) * (2 * (x - x1[i]) + h) * Runge(x1[i]) / (h ** 3)
  + ((x - x1[i]) ** 2) * (2 * (x1[i + 1] - x) + h) * Runge(x1[i + 1]) / (h ** 3)
  + ((x1[i + 1] - x) ** 2) * (x - x1[i]) * m_i / (h ** 2)
  + ((x - x1[i]) ** 2) * (x - x1[i + 1]) * m_i_next / (h ** 2)
 
  return S3

def Spline_second(x, x1):
  h = x1[1] - x1[0]
  m_i = Runge_dif(x1[0])
  m_i_next = Runge_dif(x1[1])
  
  S3 = (
  ((x1[1] - x) ** 2) * (2 * (x - x1[0]) + h) * Runge(x1[0]) / (h ** 3)
  + ((x - x1[0]) ** 2) * (2 * (x1[1] - x) + h) * Runge(x1[1]) / (h ** 3)
  + ((x1[1] - x) ** 2) * (x - x1[0]) * m_i / (h ** 2)
  + ((x - x1[0]) ** 2) * (x - x1[1]) * m_i_next / (h ** 2)
  return S3
 
x = np.linspace(-1, 1, 500)
k = [5, 8, 15, 16]
 
for jj in k:
  x1 = np.linspace(-1, 1, jj)
  
for kk in range(0, jj - 1):
  x = np.linspace(x1[kk], x1[kk + 1], round(500 / jj))
  plt.plot(x, np.array([Runge(ii) for ii in x]), color="r")
  if kk == 0:
    flager = 0
    buff = x1[:3]
  elif kk == jj - 2:
    buff = x1[-3:]
    flager = 2
  else:
    buff = x1[kk - 1 : kk + 3]
    flager = 1
    
plt.plot(x, np.array([Spline_first(ii, buff, flager) for ii in x]), color="g")
plt.plot(x, np.array([Spline_second(ii, [x1[kk], x1[kk + 1]]) for ii in x]),color="b")
plt.xlabel(r"$x$", fontsize=14)
plt.ylabel(r"$f(x)$", fontsize=14)
plt.grid(True)
plt.legend(loc="best", fontsize=12)
plt.show()
