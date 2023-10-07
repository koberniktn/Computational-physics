import numpy as np
import math
import random

def our_func(x):
  return x * math.atan(x)

def monte_carlo_v1(function, a, b, n):
  sum_of_unif_distr = [function(random.uniform(a, b)) for kk in range(n)]
  return (b - a) / n * sum(sum_of_unif_distr)

def monte_carlo_v2(function, a, b, n):
  x = np.linspace(a, b, n)
  max_f = max([function(i) for i in x])
  min_f = min([function(i) for i in x])
  uppers, downers = 0, 0
  
  for j in range(n):
    point = [random.uniform(a, b), random.uniform(0, max_f)]
    if point[1] < function(point[0]):
      downers += 1
  whole_area = (max_f) * (b - a)
  return whole_area * downers / n

def monte_carlo_v1_advanced_symm(function, a, b, n):
  sum_of_unif_distr = 0
  for kk in range(n):
    rand_num = random.uniform(a, b)
    sum_of_unif_distr += 1 / 2 * (function(rand_num) + function(b - rand_num))
  return (b - a) * sum_of_unif_distr / n

def monte_carlo_v1_advanced_mainp(function, a, b, n, helper_function):
  sum_of_unif_distr = 0
  for kk in range(n):
    rand_num = random.uniform(a, b)
    sum_of_unif_distr += (function(rand_num) - helper_function(rand_num))
  return (b - a) / n * sum_of_unif_distr

def monte_carlo_v1_advanced_distr(function, a, b, n, distribution):
  sum_of_unif_distr = 0
  for kk in range(n):
    rand_num = random.uniform(a, b)
    sum_of_unif_distr += function(rand_num) / distribution(rand_num)
  return sum_of_unif_distr / n

print('Интегрирование с помощью метода ММК с вычислением среднего')
print(f'Значение интеграла exp(x) = {monte_carlo_v1(math.exp, 0, 1, 50000)}')
print(f'Значение интеграла sin(x) = {monte_carlo_v1(math.sin, 0, math.pi/2, 50000)}')
print(f'Значение интеграла x*arctg(x) = {monte_carlo_v1(our_func, 0, math.sqrt(3), 50000)}')
print(' ')
print('Интегрирование с помощью метода ММК с интерпретацией интеграла как площади')
print(f'Значение интеграла exp(x) = {monte_carlo_v2(math.exp, 0, 1, 50000)}')
print(f'Значение интеграла sin(x) = {monte_carlo_v2(math.sin, 0, math.pi/2, 50000)}')
print(f'Значение интеграла x*arctg(x) = {monte_carlo_v2(our_func, 0, math.sqrt(3), 50000)}')
print(' ')
print('Интегрирование с помощью метода ММК с вычислением среднего и со снижением дисперсии методом симметризации')
print(f'Значение интеграла exp(x) = {monte_carlo_v1_advanced_symm(math.exp, 0, 1, 50000)}')
print(f'Значение интеграла sin(x) = {monte_carlo_v1_advanced_symm(math.sin, 0, math.pi/2, 50000)}')
print(f'Значение интеграла x*arctg(x) = {monte_carlo_v1_advanced_symm(our_func, 0, math.sqrt(3), 50000)}')
print(' ')
print('Интегрирование с помощью метода ММК с вычислением среднего и со снижением дисперсии методом главной части')
print(f'Значение интеграла exp(x) = {monte_carlo_v1_advanced_mainp(math.exp, 0, 1, 50000, lambda x: 1 + x) + 3 / 2}')
print(f'Значение интеграла sin(x) = {monte_carlo_v1_advanced_mainp(math.sin, 0, math.pi/2, 50000, lambda x: x) + (math.pi ** 2) / 8}')
print(f'Значение интеграла x*arctg(x) = {monte_carlo_v1_advanced_mainp(our_func, 0, math.sqrt(3), 50000, lambda x: x * x) + math.sqrt(3)}')
print(' ')
print('Интегрирование с помощью метода ММК с вычислением среднего и со снижением дисперсии методом существенной выборки')
print(f'Значение интеграла exp(x) = {monte_carlo_v1_advanced_distr(math.exp, 0, 1, 50000, lambda x: 2 * (1 + x) / 3)}')
print(f'Значение интеграла sin(x) = {monte_carlo_v1_advanced_distr(math.sin, 0, math.pi/2, 50000, lambda x: 8 * x/(math.pi ** 2))}')
print(f'Значение интеграла x*arctg(x) = {monte_carlo_v1_advanced_distr(our_func, 0, math.sqrt(3), 50000, lambda x: x * x/math.sqrt(3))}')
