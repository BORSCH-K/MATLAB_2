import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt


def shift(a, n0):
    t = a.copy()
    for i in range(0, len(a) - n0):
        a[i + n0] = t[i]
        # print(i, t[i])
        # print(a)
    j = 0
    for i in range(len(a) - n0, len(a)):
        a[j] = t[i]
        # print(i, j, t[i])
        # print(a)
        j += 1
    # print(a)
    return a


def main():
    fd = 1024
    f = 16
    ph = 0
    N = fd / f
    A = 1
    n = np.arange(0, N)

    a = 2
    # y = A * np.sin(2 * np.pi * n / N + ph)
    # y = A * sg.square(2 * np.pi * n / N + ph, duty=0.5)
    # x = y * a  # тот же сигнал у, но увеличенный
    # x = A * sg.square(2 * np.pi * n / N + ph, duty=0.25)
    x = A * np.sin(2 * np.pi * n / N + ph)
    y = shift(x.copy(), 2)



    # print(x)
    # print(y)

    X = np.fft.fft(x)
    X_a = abs(X)
    Y = np.fft.fft(y)
    Y_a = abs(Y)

    # проверка маштабируемости уровня сигнала
    k = 0
    for i in range(0, len(Y)):
        if Y[i] * a == X[i]:
            k += 1
            # print("true")
        # else:
        # print("false")
    if k == len(Y):
        print("true")  # сигналы совпали
    else:
        print("false")

    z = x + y
    print(z)

    Z = np.fft.fft(z)
    print(X)
    print(Y)
    print(Z)

    print(x)
    print(y)

    Zk = X + Y
    print(Zk)

    fig, ax = plt.subplots()
    ax.plot(n / fd, x)
    ax.plot(n / fd, y)
    ax.plot(n / fd, z)
    ax.grid()
    ax.set_xlabel('t,c')
    ax.set_ylabel('A')

    fig0, ax0 = plt.subplots()
    ax0.plot(n / fd, Z+1)
    ax0.plot(n / fd, Zk)
    ax0.grid()
    ax0.set_xlabel('t,c')
    ax0.set_ylabel('A')

    fig1, ax1 = plt.subplots()
    ax1.plot(n / fd, x)
    ax1.plot(n / fd, y)
    ax1.grid()
    ax1.set_xlabel('t,c')
    ax1.set_ylabel('A')

    fig2, ax2 = plt.subplots()
    ax2.stem(n * f, X_a)
    ax2.plot(n * f, Y_a, 'ro')
    ax2.grid()
    ax2.set_xlabel('w,Гц')
    ax2.set_ylabel('Aмплитуда')

    FSx = np.angle(X)
    FSy = np.angle(Y)

    fig3, ax3 = plt.subplots()
    ax3.plot(n * f, FSx)
    ax3.plot(n * f, FSy)
    ax3.grid()
    ax3.set_xlabel('w,Гц')
    ax3.set_ylabel('Сдвиг фазы')

    tm = []
    for i in range(0, len(x), 2):
        tm.append(x[i])
    print(len(tm))
    fig4, ax4 = plt.subplots()
    ax4.stem(n * f, tm + tm)
    ax4.plot(n * f, x, 'ro')
    ax4.grid()
    ax4.set_xlabel('w,Гц')
    ax4.set_ylabel('Aмплитуда')

    TM = np.fft.fft(tm + tm)
    TMa = abs(TM)
    print(len(TMa))

    fig5, ax5 = plt.subplots()
    ax5.stem(n * f, TMa)
    ax5.plot(n * f, X_a, 'ro')
    ax5.grid()
    ax5.set_xlabel('w,Гц')
    ax5.set_ylabel('Aмплитуда')

    plt.show()


if __name__ == '__main__':
    main()
