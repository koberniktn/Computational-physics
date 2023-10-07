import numpy as np
import matplotlib.pyplot as plt

def answer(x):
  return np.exp(x ** 2) / x

def answer_task(x):
  return np.exp(-0.1 * x) * (np.cos(np.sqrt(0.99) * x) + 0.1 / np.sqrt(0.99) * np.sin(np.sqrt(0.99) * x))

def u_diff(x, u):
  return 2 * np.exp(x ** 2) - u / x

def Eulers_method(x):
  u = [np.exp(1)]
  h = x[1] - x[0]
  for i in range(1, len(x)):
  u.append(u[i - 1] + h * u_diff(x[i - 1], u[i - 1]))
  return u

def Eulers_method_modified(x):
  u = [np.exp(1)]
  h = x[1] - x[0]
  for i in range(1, len(x)):
  u.append(u[i - 1] + (h / 2) * (u_diff(x[i - 1], u[i - 1]) + u_diff(x[i], u[i - 1] + h * u_diff(x[i - 1], u[i - 1]))))
  return u

def Eulers_method_advanced(x):
  u = [np.exp(1)]
  h = x[1] - x[0]
  for i in range(1, len(x)):
  u.append(u[i - 1] + h * u_diff(x[i - 1] + h / 2, u[i - 1] + h / 2 * u_diff(x[i - 1], u[i - 1])))
  return u

def Rhunge_Khuttas_method(x):
  u = [np.exp(1)]
  h = x[1] - x[0]
  for i in range(1, len(x)):
  k_0 = u_diff(x[i - 1], u[i - 1])
  k_1 = u_diff(x[i - 1] + h / 2, u[i - 1] + h * k_0 / 2)
  k_2 = u_diff(x[i - 1] + h / 2, u[i - 1] + h * k_1 / 2)
  k_3 = u_diff(x[i - 1] + h, u[i - 1] + h * k_2)
  u.append(u[i - 1] + (h / 6) * (k_0 + 2 * k_1 + 2 * k_2 + k_3))
  return u

def task_9(x):
  u = [1]
  diff_u = [0]
  h = x[1] - x[0]
  for i in range(1, len(x)):
  diff_u.append(diff_u[i - 1] + h * (- u[i - 1] - 0.2 * diff_u[i - 1]))
  u.append(u[i - 1] + h * diff_u[i])
  return u

x = np.linspace(1, 4, 500)
x_for_task = np.linspace(0, 12.63, 500)

plt.plot(x, answer(x), color='g', label='Аналитическое решение')
plt.plot(x, Eulers_method(x), color='r', label='Метод Эйлера')
plt.plot(x, Eulers_method_modified(x), color='b', label='Модифицированный метод Эйлера')
plt.plot(x, Eulers_method_advanced(x), color='orange', label='Усовершенствованный метод Эйлера')
plt.plot(x, Rhunge_Khuttas_method(x), color='black', label='Метод Рунге-Кутта')
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$f(x)$', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.show()

plt.plot(x_for_task, answer_task(x_for_task), color='green', label='Аналитическое решение')
plt.plot(x_for_task, task_9(x_for_task), color='black', label='Численное решение')
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$f(x)$', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.show()
