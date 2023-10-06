import math
import numpy as np
import plotly.graph_objects as go

def u_start(x):
    res = [1+math.cos(3*i/4) for i in x]
    return res

def graphic(x,t,u):
    fig = go.Figure(go.Surface(
        x=x,
        y=t,
        z=u))
    fig.show()

def check(tau,h,a):
    if tau < (2 * ((h / a) ** 2)):
        return True
    else:
        return False

def eq_heat_explicit(tau, h, a, l_max, t_max):
    if check(tau, h, a):
        t = np.linspace(0, t_max, round(t_max / tau))
        x = np.linspace(0, l_max, round(l_max / h))
        u = [[0 for i in range(len(x))] for j in range(len(t))]
        u[0] = u_start(x)
        for i in range(0, len(t)-1):
            for j in range(1, len(x)-1):
                u[i+1][j] = u[i][j] + tau * (a ** 2) * (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (h ** 2) + math.cos(x[j]) * tau
            u[i + 1][0] = 0
            u[i + 1][len(x)-1] = u[i + 1][len(x)-2]
        graphic(x, t, u)
    else:
        print("Чето плохо идет")

def eq_heat_not_explicit(tau, h, a, l_max, t_max):
    if check(tau, h, a):
        t = np.linspace(0, t_max, round(t_max / tau))
        x = np.linspace(0, l_max, round(l_max / h))
        u = [[0 for i in range(len(x))] for j in range(len(t))]
        u[0] = u_start(x)
        alpha = 1
        beta = - 2 - (h ** 2) / ((a ** 2) * tau)
        gamma = 1
        A, B = [0], [0]
        for i in range(1, len(t)):
            for j in range(1, len(x) - 1):
                delta = - u[i - 1][j] * (h ** 2) / ((a ** 2) * tau) - (h ** 2) * math.cos(x[j]) / (a ** 2)
                B.append((delta - gamma * B[-1]) / (beta + gamma * A[-1]))
                A.append(- alpha / (beta + gamma * A[-1]))
            u[i][-1] = B[-1] / (1 - A[-1])
            for k in range(1, len(x) - 1):
                u[i][-k - 1] = u[i][-k] * A[-k] + B[-k]
            u[i][0] = 0
            u[i][-2] = u[i][-1]
        graphic(x, t, u)
    else:
        print("Чето плохо идет")


status = 'устойчивые'
tau = 0.01
h = 0.1
a = 0.3
l_max = 2*math.pi
t_max = 1
eq_heat_not_explicit(tau, h, a, l_max, t_max)
