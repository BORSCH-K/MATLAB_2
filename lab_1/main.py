import numpy as np
import scipy as sp
from scipy import signal as sg
import matplotlib.pyplot as plt


def main():
    fd = 1024  # частота дискретизации; сколько раз за секунду измеряется моментальное значение сигнала
    f = 16  # частота сигнала; длина шага
    ph = 0  # начальная фаза
    N = fd / f  # число точек на периоде
    A = 1  # амплитуда
    n = np.arange(0, N)  # номер шага
    y = A * np.sin(2 * np.pi * n / N + ph)

    fig, ax = plt.subplots()
    ax.plot(n / fd, y)  # период сигнала = 0,0625
    ax.grid()
    ax.set_xlabel('t,c')
    ax.set_ylabel('A')

    # Скважность
    fig2, ax2 = plt.subplots()
    y1 = A * sg.square(2 * np.pi * n / N + ph, duty=0.5)
    ax2.plot(n / fd, y1)
    ax2.grid()
    ax2.set_xlabel('t,c')
    ax2.set_ylabel('A')

    fig3, ax3 = plt.subplots()
    y2 = A * sg.square(2 * np.pi * n / N + ph, duty=0.25)
    ax3.plot(n / fd, y2)
    ax3.grid()
    ax3.set_xlabel('t,c')
    ax3.set_ylabel('A')

    # ПД, АЧХ, ФЧХ

    CS = np.fft.fft(y)

    fig4, ax4 = plt.subplots()
    AS = np.abs(CS)
    ax4.stem(n * f, AS)
    ax4.grid()
    ax4.set_xlabel('w,Гц')
    ax4.set_ylabel('Aмплитуда')

    fig5, ax5 = plt.subplots()
    FS = np.angle(CS)
    ax5.plot(n * f, FS)
    ax5.grid()
    ax5.set_xlabel('w,Гц')
    ax5.set_ylabel('Сдвиг фазы')

    CS1 = np.fft.fft(y1)

    fig6, ax6 = plt.subplots()
    AS1 = np.abs(CS1)
    ax6.stem(n * f, AS1)
    ax6.grid()
    ax6.set_xlabel('w,Гц')
    ax6.set_ylabel('Aмплитуда')

    fig7, ax7 = plt.subplots()
    FS1 = np.angle(CS1)
    ax7.plot(n * f, FS1)
    ax7.grid()
    ax7.set_xlabel('w,Гц')
    ax7.set_ylabel('Сдвиг фазы')

    CS2 = np.fft.fft(y2)

    fig8, ax8 = plt.subplots()
    AS2 = np.abs(CS2)
    ax8.stem(n * f, AS2)
    ax8.grid()
    ax8.set_xlabel('w,Гц')
    ax8.set_ylabel('Aмплитуда')

    fig9, ax9 = plt.subplots()
    FS2 = np.angle(CS1)
    ax9.plot(n * f, FS2)
    ax9.grid()
    ax9.set_xlabel('w,Гц')
    ax9.set_ylabel('Сдвиг фазы')

    # востановление графиков
    z = np.fft.ifft(CS)
    # z1 = np.ifft(CS1)
    # z2 = np.ifft(CS2)

    fig10, ax10 = plt.subplots()
    ax10.plot(n / fd, z)
    ax10.grid()
    ax10.set_xlabel('t,c')
    ax10.set_ylabel('A')

    # 2Т

    n_2 = np.arange(0, 2 * N)
    y_2 = A * np.sin(2 * np.pi * n_2 / N + ph)

    fig11, ax11 = plt.subplots()
    ax11.plot(n_2 / fd, y_2)
    ax11.grid()
    ax11.set_xlabel('t,c')
    ax11.set_ylabel('A')

    CS_2 = np.fft.fft(y_2)
    AS_2 = np.abs(CS_2)
    fig12, ax12 = plt.subplots()
    ax12.stem(n_2 * f/2, AS_2)
    ax12.grid()
    ax12.set_xlabel('w,Гц')
    ax12.set_ylabel('Aмплитуда')

    plt.show()


if __name__ == '__main__':
    main()
