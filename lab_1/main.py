from scipy.integrate import solve_ivp
import numpy as np
import math
from matplotlib import pyplot as plt

#9 вариант

C = 0.15  # баллистический коэффициент

rho_lead = 7874  # kg/m^3 - плотность железа
rho_air = 1.29  # kg/m^3 - плотность воздуха

v0_1 = 10  # начальная скорость

rad = 0.09  # радиус шарика

t_0 = 0  # время начала
t_max = 100  # время конца

S = math.pi * (rad ** 2)  # площадь поперечного сечения
beta = C * rho_air * S / 2
V = (4 / 3) * math.pi * (rad ** 3)  # объем шара
m = rho_lead * V  # масса шара

eps = 1.e-2
g = 9.8  # m/sec^2


# Начальные условия


# Галилей:
def x(t, alph, v00):
    return v00 * math.cos(alph) * t


def y(t, alph, v00):
    return v00 * math.sin(alph) * t - g * (t ** 2) / 2


# Ньютон:
def right_part(t, system):
    (u, w, x, y) = system
    factor = -beta * math.sqrt(u ** 2 + w ** 2) / m
    return np.ndarray((4,), buffer=np.array([u * factor, w * factor - g, u, w]))


# удаляем все точки, где x < 0
def trim(arr):
    M = np.where(arr[1] >= 0)[-1][-1]
    return arr[:M, :M]


if __name__ == "__main__":

    alpha = math.radians(34)  # угол в радианах
    v0 = 125  # тестирование начальной скорости

    data_newton = []
    data_gal = []

    for speed in np.arange(10, 125.0, 10.0):
        v0 = speed

        u0 = v0 * math.cos(alpha)
        w0 = v0 * math.sin(alpha)
        x0 = 0
        y0 = 0

        coords = solve_ivp(right_part, (t_0, t_max), np.ndarray((4,), buffer=np.array([u0, w0, x0, y0])), max_step=eps)

        #print(coords)
        t_arr = coords['t']
        coords = coords['y'][2:]

        coords = trim(coords)

        xvals, yvals = coords

        gal_xvals = [x(t, alpha, v0) for t in t_arr]
        gal_yvals = [y(t, alpha, v0) for t in t_arr]

        gal_coords = np.ndarray((2, len(gal_xvals)), buffer=np.array([gal_xvals, gal_yvals]))
        gal_coords = trim(gal_coords)
        gal_xvals, gal_yvals = gal_coords

        data_gal.append((gal_xvals, gal_yvals))
        data_newton.append((xvals, yvals))

        #print("Galilei: (", gal_xvals[-1], ", ", gal_yvals[-1], ")", sep='')
        #print("Newton : (", xvals[-1], ", ", yvals[-1], ")", sep='')

        #plt.plot(gal_xvals, gal_yvals, 'r-', label='Galileo model')
        #plt.plot(xvals, yvals, 'b', label='Newton model')

    for i in range(len(data_newton)):
        plt.plot(data_gal[i][0], data_gal[i][1], 'r-', label=f'Galileo model {i}')
        plt.plot(data_newton[i][0], data_newton[i][1], 'b', label=f'Newton model {i}')

    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
