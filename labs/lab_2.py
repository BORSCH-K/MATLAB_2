import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt


def energy(array, num):
    e = 0.
    for i in range(0, num):
        e += (abs(array[i]) ** 2) / num
    return e


def energy_wane(array, i, e0, num):
    i1 = i  # left
    i2 = i  # right
    e = e0
    e_temp = e0
    array_temp = array
    while e_temp >= e0 * 0.95:
        array = array_temp
        e = e_temp
        array_temp[i1] = 0
        array_temp[i2] = 0
        i1 -= 1
        i2 += 1
        e_temp = energy(array_temp, num)
    while array[i1] == 0:
        i1 -= 1
    print(i1)  # , array[i1])
    return array, e


def main():
    fd = 1024
    f = 16
    ph = 0
    N = fd / f
    A = 1
    n = np.arange(0, N)

    y = A * np.sin(2 * np.pi * n / N + ph)
    y1 = A * sg.square(2 * np.pi * n / N + ph, duty=0.5)
    y2 = A * sg.square(2 * np.pi * n / N + ph, duty=0.25)

    CS = np.fft.fft(y)
    AS = np.abs(CS)
    E0 = energy(AS, int(N))
    # print(E0)

    CS1 = np.fft.fft(y1)
    AS1 = np.abs(CS1)
    E1 = energy(AS1, int(N))
    # print(E1)

    AS1_n, E1_n = energy_wane(AS1, 32, E1, int(N))
    # print(AS1_n, E1_n)

    fig1, ax1 = plt.subplots()
    ax1.stem(n * f, AS1_n)
    ax1.grid()
    ax1.set_xlabel('w,Гц')
    ax1.set_ylabel('Aмплитуда')

    CS2 = np.fft.fft(y2)
    AS2 = np.abs(CS2)
    E2 = energy(AS2, int(N))
    # print(E2)

    AS2_n, E2_n = energy_wane(AS2, 32, E2, int(N))
    # print(AS2_n, E2_n)

    fig2, ax2 = plt.subplots()
    ax2.stem(n * f, AS2_n)
    ax2.grid()
    ax2.set_xlabel('w,Гц')
    ax2.set_ylabel('Aмплитуда')

    # plt.show()


    for q in range(1, 99):
        y0 = A * sg.square(2 * np.pi * n / N + ph, duty=q / 100)
        CS0 = np.fft.fft(y0)
        AS0 = np.abs(CS0)
        E0 = energy(AS0, int(N))
        print(q, ":")
        AS0_n, E0_n = energy_wane(AS0, 32, E0, int(N))


if __name__ == '__main__':
    main()
