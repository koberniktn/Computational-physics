from numpy import random
import math

def function1(r):
  return 1 / r

def function2(r):
  return 1 / (r ** 2)
def function3(r):
  return r

def for_ideal_functions(function, a, b, n):
  sum = 0
  for i in range(n):
    r_p, r_q, theta = random.uniform(a, b), random.uniform(a, b), random.uniform(-b, b)
    r = (r_p ** 2 + r_q ** 2 - 2 * theta * r_p * r_q) ** (1 / 2)
  sum += function(r)
  return 16 * sum / (n * 9)

def for_non_ideal_functions(function, a, b, n):
  if function == function2:
    parameter = 1 / 3
  else:
    parameter = 1
  sum = 0
  
  for i in range(n):
    r_p, r_q, theta = random.uniform(a, b) ** (parameter), random.uniform(a, b), 2 * random.uniform(a, b) - 1
    r = ((r_p ** 2) + (r_q ** 2) - 2 * theta * r_p * r_q) ** (1 / 2)
    l = r_p * theta + (1 - (r_p ** 2) * (1 - theta ** 2)) ** (1 / 2)
    sum += ((r ** 2) * function(r) * l)
  return 16 * sum / (n * 3)

print(f'Метод вычисления среднего при m = 1: {for_ideal_functions(function1, 0, 1, 100)}')
print(f'Метод вычисления среднего при m = 2: {for_ideal_functions(function2, 0, 1, 100)}')
print(f'Метод вычисления среднего при m = -1: {for_ideal_functions(function3, 0, 1, 20000)}')

print(f'Метод вычисления среднего c особенностью при m = 1: {for_non_ideal_functions(function1, 0, 1, 50000)}')
print(f'Метод вычисления среднего c особенностью при m = 2: {for_non_ideal_functions(function2, 0, 1, 50000)}')
print(f'Метод вычисления среднего c особенностью при m = -1: {for_non_ideal_functions(function3, 0, 1, 50000)}')
