import numpy as np
import matplotlib.pyplot as plt
import math

def function(x):
  return x * math.atan(x)

def rectangle_method(x):
  sum = 0
  for i in range(len(x)-1):
    sum += (x[i+1] - x[i]) * function((x[i+1] + x[i]) / 2)
  return sum

def trapezoid_method(x):
  h = x[1] - x[0]
  sum = 0
  for i in range(len(x)-1):
    sum += (function(x[i+1]) + function(x[i])) / 2 * h
  return sum

def Simpson_method(x):
  h = x[1] - x[0]
  sum = 0
  for i in range(round((len(x)-2)/2)):
    sum += h / 3 * (function(x[2 * i]) + 4 * function(x[2 * i + 1]) + function(x[2 * (i + 1)]))
  return sum

def Richardson_method(x, integration_method):
  if integration_method in [trapezoid_method, rectangle_method]:
    p = 2
  else:
    p = 3
  h = x[1] - x[0]
  x0 = np.linspace(0, math.sqrt(3), round((2 * math.sqrt(3) / h)))
  int_cur = integration_method(x)
  int_next = integration_method(x0) + (integration_method(x0) - integration_method(x)) / (2 ** p - 1)
  eps = h ** p
  while abs(int_next - int_cur) > eps:
    x = x0
    h /= 2
    x0 = np.linspace(0, math.sqrt(3), round((2 * math.sqrt(3)/h)))
    int_cur = int_next
    int_next = integration_method(x0) + (integration_method(x0) - integration_method(x)) / (2 ** p - 1)
  return int_next
  
	# должно быть четное кол-во отрезков в сетке
for length in [500, 1000 ,1500]:
  x = np.linspace(0, math.sqrt(3), length)
  
print(f'Количество точек = {length}')
print(f'Интегрирование методом прямоугольников: I = {rectangle_method(x)}')
print(f'Интегрирование методом трапеций: I = {trapezoid_method(x)}')
print(f'Интегрирование методом Симпсона: I = {Simpson_method(x)}')
print(f'Интегрирование методом прямоугольников + экстраполяция Ричардсона: I = {Richardson_method(x, rectangle_method)}')
print(f'Интегрирование методом трапеций + экстраполяция Ричардсона: I = {Richardson_method(x, trapezoid_method)}')
print(f'Интегрирование методом Симпсона + экстраполяция Ричардсона: I = {Richardson_method(x, Simpson_method)}')
