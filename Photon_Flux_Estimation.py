import math
import numpy as np
import random
import matplotlib.pyplot as plt

def disk(x, a):
    return (a ** 2 - x ** 2) ** (1 / 2)

def p_x(x, a):
    p = (x / a) + (a / x) + ((1 / a) - (1 / x)) * (2 + (1 / a) - (1 / x))
    return p

def sigma_E_calk(a, h):
    A = Z_num * (r_electron ** 2)
    B = 1 - h
    C = 1 + a
    D = 2 * a + 1
    s_1 = A / 2 * ((1 + a * B) ** (-2)) * (1 + (h ** 2) + ((a ** 2) * (B ** 2)) / (1 + a * B))
    s_2 = 2 * math.pi * A * (
                (C / (a ** 2)) * ((2 * C / D) - (math.log(D) / a)) + (math.log(D) / (2 * a)) - ((1 + 3 * a) / (D ** 2)))
    return [s_1, s_2]

def random_points_3(R):
    gamma_1 = random.uniform(0, 1)
    gamma_2 = random.uniform(0, 1)
    r = R * (gamma_1 ** (1 / 2))
    psi = 2 * math.pi * gamma_2
    return [r * math.cos(psi), r * math.sin(psi)]

def random_angles():
    gamma_3 = random.uniform(0, 1)
    cos_tetta = 2 * gamma_3 - 1
    d = 2
    while (d > 1):
        gamma_4 = random.uniform(0, 1)
        gamma_5 = random.uniform(0, 1)
        a = 2 * gamma_4 - 1
        b = 2 * gamma_5 - 1
        d = ((a ** 2) + (b ** 2))
    cos_phi = a / (d ** (1 / 2))
    sin_phi = b / (d ** (1 / 2))
    return [sin_phi, cos_phi, cos_tetta]

def sigma_count(E):
    energy_array = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    sigma = [-1, -1, -1]  # вышел за пределы энергий

    if ((E >= energy_array[0]) & (E <= energy_array[17])):
        for i in range(len(energy_array)):
            buf = energy_array[i]
            if (E == buf):
                sigma[0] = SIGMA[i]
                sigma[1] = sig_comp[i]
                sigma[2] = sig_ph[i]
            elif (E < buf):
                sigma[0] = SIGMA[i - 1] + (E - energy_array[i - 1]) * (SIGMA[i] - SIGMA[i - 1]) / (
                            buf - energy_array[i - 1])
                sigma[1] = sig_comp[i - 1] + (E - energy_array[i - 1]) * (sig_comp[i] - sig_comp[i - 1]) / (
                            buf - energy_array[i - 1])
                sigma[2] = sig_ph[i - 1] + (E - energy_array[i - 1]) * (sig_ph[i] - sig_ph[i - 1]) / (
                            buf - energy_array[i - 1])
    return sigma

def run_lenght(sigma):
    gamma_6 = random.uniform(0, 1)
    L = -math.log(gamma_6) / sigma
    return L

def check_position(r, l):
    if ((-l / 2 <= r[0] <= l / 2) & (-l / 2 <= r[1] <= l / 2) & (0 <= r[2] <= l)):
        return 1
    else:
        return 0

def interaction_type(s):
    sum = s[0] / P
    gamma_7 = random.uniform(0, 1)
    if (gamma_7 < (s[1] / sum)):
        return 1
    elif (gamma_7 < ((s[1] + s[2]) / sum)):
        return 2
    else:
        return 0

def random_energy(a):
    p = 1
    coef = 0
    x = 0
    while (p >= coef):
        gamma_1 = random.uniform(0, 1)
        gamma_2 = random.uniform(0, 1)

        x = (a * (1 + 2 * a * gamma_1)) / (1 + 2 * a)
        p = p_x(x, a)
        coef = gamma_2 * (1 + 2 * a + (1 / (1 + 2 * a)))
    return x

def energy_group(E):
    K = 0
    if 0.02 <= E <= 0.03:
        K = 1
    elif 0.03 < E <= 0.04:
        K = 2
    elif 0.04 < E <= 0.05:
        K = 3
    elif 0.05 < E <= 0.06:
        K = 4
    elif 0.06 < E <= 0.08:
        K = 5
    elif 0.08 < E <= 0.1:
        K = 6
    elif 0.1 < E <= 0.15:
        K = 7
    elif 0.15 < E <= 0.2:
        K = 8
    elif 0.2 < E <= 0.3:
        K = 9
    elif 0.3 < E <= 0.4:
        K = 10
    elif 0.4 < E <= 0.5:
        K = 11
    elif 0.5 < E <= 0.6:
        K = 12
    elif 0.6 < E <= 0.8:
        K = 13
    elif 0.8 < E <= 1.0:
        K = 14
    elif 1.0 < E <= 1.5:
        K = 15
    elif 1.5 < E <= 2.0:
        K = 16
    elif 2.0 < E <= 3.0:
        K = 17
    return (K - 1)

def W_count(angle):
    W = [0] * 3
    W[0] = angle[1] * angle[2]
    W[1] = angle[0] * angle[2]
    W[2] = pow((1 - pow(angle[2], 2)), 1 / 2)
    return W

def W_recount(W, angle):
    W_res = [0] * 3

    q = ((1 - (angle[2] ** 2)) * (1 - (W[2] ** 2))) ** (1 / 2)
    W_res[2] = W[2] * angle[2] + q * angle[1]

    a = angle[2] - W[2] * W_res[2]
    b = angle[0] * q
    c = 1 - (W[2]) ** 2

    W_res[1] = ((W[1] * a) + (W[0] * b)) / c
    W_res[0] = ((W[0] * a) - (W[1] * b)) / c
    return W_res

def eta_calc(r, cos_tetta, a_new, W, Sigma):
    eta = [0, 0, 0, 0, 0, 0, 0]
    Sensor = [[0] * 3] * 7
    Sensor[0] = [10, -10, 0]
    Sensor[1] = [10, -10, 5]
    Sensor[2] = [10, -10, 20]
    Sensor[3] = [10, 0, 20]
    Sensor[4] = [10, 10, 20]
    Sensor[5] = [0, 0, 0]
    Sensor[6] = [0, 0, 5]

    sigma_E = sigma_E_calk(a_new, cos_tetta)
    scat_indicatrix = W * sigma_E[0] / sigma_E[1]

    for i in range(len(eta)):
        delta_r = ((r[0] - Sensor[i][0]) ** 2) + ((r[1] - Sensor[i][1]) ** 2) + ((r[2] - Sensor[i][2]) ** 2)
        eta[i] = scat_indicatrix * math.exp(- Sigma * (delta_r ** (1 / 2))) / delta_r

    return eta

sig_comp = (0.147, 0.142, 0.138, 0.134, 0.130, 0.123,
            0.117, 0.106, 0.0968, 0.0843, 0.0756, 0.0689,
            0.0637, 0.0561, 0.0503, 0.0410, 0.0349, 0.0274)
sig_ph = (83.1, 28.5, 13.2, 7.21, 4.39,
          1.97, 5.23, 1.8, 0.843, 0.289,
          0.141, 0.0823, 0.0538, 0.0285, 0.0180, 0.00858, 0.00523, 0.00282)
sig_pair = (0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0.00159, 0.005, 0.0115)

SIGMA = [0] * len(sig_comp)

P = 11.35  # плотность свинца
Z_num = 82  # атомный номер свинца
r_electron = 2.8 * (10 ** (-13))  # радиус электрона

for i in range(len(SIGMA)):
    SIGMA[i] = (sig_comp[i] + sig_ph[i] + sig_pair[i]) * P

x = np.linspace(-5, 5, 500)
plt.grid(True)
plt.figure(1)
plt.plot(x, disk(x, 5), color='r')
plt.plot(x, -disk(x, 5), color='r')

for k in range(500):
    point = random_points_3(5)
    plt.plot(point[0], point[1], 'go', markersize=2)  # график разброса точек на плоскости

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, disk(x, 5), color='r')
ax.plot(x, -disk(x, 5), color='r')

x = np.linspace(-10, 10, 10)
y = np.linspace(-10, 10, 10)
z = np.linspace(0, 20, 10)

values = [20, 10, -10, 10, -10]

for i in range(5):
    if (i == 0):
        X, Y = np.meshgrid(x, y)
        eq = 20 * (X - X) + values[i]
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, eq, rcount=1, color='blue', alpha=0.3)
    elif ((i > 0) & (i < 3)):
        Y, Z = np.meshgrid(y, z)
        eq = 20 * (Y - Y) + values[i]
        ax = fig.gca(projection='3d')
        ax.plot_surface(eq, Y, Z, rcount=1, color='blue', alpha=0.3)
    elif (i >= 3):
        X, Z = np.meshgrid(x, z)
        eq = 20 * (X - X) + values[i]
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, eq, Z, rcount=1, color='blue', alpha=0.3)

ph = 0
pair = 0
fly = 0
small_energy = 0
small_weight = 0
n = 100
R = [0, 0]
counter = 0
counter_sum = 0
PHI = np.empty((7, 18))
eta = np.empty((7, 18))

for j in range(n):
    for l in range(7):
        for q in range(17):
            PHI[l][q] += eta[l][q]
    eta = np.empty((7, 18))
    print("PHI = ", PHI)

    W_stat = 1  # статистический вес частицы
    counter_sum += counter

    r_0 = [random_points_3(5)[0], random_points_3(5)[1], 0]
    R[0] = r_0
    angle = random_angles()

    energy_old = 3.0  # н.у.
    K = 17  # номер энергетической группы
    sigma = sigma_count(energy_old)
    L_1 = run_lenght(sigma[0])

    W = W_count(angle)
    r_1 = [r_0[0] + L_1 * W[0], r_0[1] + L_1 * W[1], L_1 * W[2]]
    R[1] = r_1

    counter = 1
    kost = 1

    while (kost != 0):
        check = check_position(R[1], 20)
        if (check == 1):
            ax.plot([R[0][0], R[1][0]], [R[0][1], R[1][1]], [R[0][2], R[1][2]], 'g-')
        else:
            fly += 1
            ax.plot([R[0][0], R[1][0]], [R[0][1], R[1][1]], [R[0][2], R[1][2]], 'r-')
            kost = 0
            continue
        inter = interaction_type(sigma)
        if inter == 0:
            pair += 1
            kost = 0
            continue
        if inter == 2:
            ph += 1
            kost = 0
            continue

        energy_new = random_energy(energy_old)
        if (energy_new <= 0.02):
            small_energy += 1
            kost = 0
            continue
        K = energy_group(energy_new)  # номер энергетической группы

        sigma = sigma_count(energy_new)  # [0] - SIGMA, [1] - sigma_comp
        W_stat = W_stat * sigma[1] / sigma[0]  # статистический вес частицы
        if (W_stat < (10 ** (-11))):
            small_weight += 1
            kost = 0
            continue
        angle_scat = [random_angles()[0], random_angles()[1], (1 + (1 / energy_old) - (
                    1 / energy_new))]  # это всё прекрасно, но косинус получается больше 1. у хасана знаки перепутаны были!

        # оценка потока
        eta_new = eta_calc(R[1], angle_scat[2], energy_new, W_stat, sigma[0])
        for f in range(7):
            eta[f][K] += eta_new[f]  # записываем отдельно вклад от каждой энергетической группы

        # дальнейшая траетория частицы
        W = W_count(angle_scat)
        W = W_recount(W, angle_scat)  # вернулись в изначальную СК

        L_2 = L_1 = run_lenght(sigma[0])
        r_2 = [r_1[0] + L_2 * W[0], r_1[1] + L_2 * W[1], r_1[2] + L_2 * W[2]]
        R[0] = R[1]
        R[1] = r_2
        counter += 1
for k in range(7):
    for l in range(17):
        PHI[k][l] = round(PHI[k][l] / n * (10 ** 9), 1)

print('\nВас приветствует Hedgehog (версия 2.1)\nСтатистика на сегодня:')
print("\nРодилось - ", n, "частиц")
print("Вылетело за пределы области - ", fly, "частиц (", round(fly / n * 100, 2), "% )")
print("Потеряли слишком много энергии - ", small_energy, "частиц (", round(small_energy / n * 100, 2), "% )")
print("Недостаточный стат. вес - ", small_weight, "частиц (", round(small_weight / n * 100, 2), "% )")
# print("Исчезло из-за взаимодействий- ", die, "частиц (", round(die/n * 100, 2), "% )\n")
print("Исчезло из-за фотоэффекта- ", ph, "частиц (", round(ph / n * 100, 2), "% )")
print("Исчезло из-за образ.пары- ", pair, "частиц (", round(pair / n * 100, 2), "% )\n")
print("В среднем частица переживает", round(counter_sum / n, 2), "столкновений")

# построение гистограммы
labels = ['0.02', '0.03', '0.04', '0.05', '0.06', '0.08', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8',
          '1.0', '1.5', '2.0', '3.0']
x_f = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
x_g = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
for k in range(7):
    fig = plt.figure(2 + k)
    fig_g, ax = plt.subplots(figsize=(8,4))
    ax.bar(x_g, PHI[k], align='edge', edgecolor='black', width=1)
    ax.set_xticks(x_g)
    ax.set_xticklabels(labels)
    fig_g.tight_layout()
    ax.set_xlabel('МэВ')
    ax.set_ylabel('10^(-3) эВ/м^2')
    ax.set_title('Поток для датчика №' + str(k+1))

print('\nХорошего дня ;)')
plt.show()
