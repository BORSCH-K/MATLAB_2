import numpy as np
from scipy import signal as sg


def energy_x(array):
    e = 0.
    for i in range(0, len(array)):
        e += array[i] ** 2
    return e


def energy_ASH(array):
    e = 0.
    for i in range(0, len(array)):
        e += (abs(array[i]) ** 2)
    return e / len(array)


def main():
    fd = 1024
    f = 16
    ph = 0
    N = fd / f
    A = 1
    n = np.arange(0, N * 2)

    y = A * np.sin(2 * np.pi * n / N + ph)
    y1 = A * sg.square(2 * np.pi * n / N + ph, duty=0.5)
    y2 = A * sg.square(2 * np.pi * n / N + ph, duty=0.25)

    CS = np.fft.fft(y)
    AS = np.abs(CS)
    Ex = energy_x(y)
    EASH = energy_ASH(AS)
    print(Ex, EASH)

    CS1 = np.fft.fft(y1)
    AS1 = np.abs(CS1)
    Ex1 = energy_x(y1)
    EASH1 = energy_ASH(AS1)
    print(Ex1, EASH1)

    CS2 = np.fft.fft(y2)
    AS2 = np.abs(CS2)
    Ex2 = energy_x(y2)
    EASH2 = energy_ASH(AS2)
    print(Ex2, EASH2)


if __name__ == '__main__':
    main()
