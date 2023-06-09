import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg

def awgn1(S, SNR):
    n = len(S)                      # число отсчетов в сигнале
    Es = np.sum(S**2) / n           # среднеквадратичное значение сигнала
    En = Es * 10 ** (-SNR / 20)     # среднеквадратичное значение шума
    WGN = np.random.randn(n) * En
    S1 = S + WGN
    return S1

def main():

    fsig = 1e4                  # частота первой гармоники сигнала
    N = 100                     # число отсчетов характеристики + 1 (нулевая гармоника)
    fs = fsig * N               # частота дискретизации
    f = np.arange(0, fs + fsig, fsig)  # диапазон чаcтот для АЧХ сигнала

    l = 1000                                    # длина линии связи, м
    R = 5e-3 + (42e-3) * np.sqrt(f * (1e-6))    # погонное сопротивление
    L = 2.7e-7                                  # погонная индуктивность
    G = 20 * f * (1e-15)                        # погонная проводимость
    C = 48e-12                                  # погонная емкость

# Задание 1
# ___________построние АЧХ и ФЧХ линии связи_________________
    w = 2 * np.pi * f                                   # вектор круговых частот
    g1 = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))   # коэффициент распространения волны
    K1 = np.exp(-g1 * l)                                # комплексная частотная характеристика линии связи

    ACH = np.abs(K1)                # АЧХ линии связи
    FCH = np.unwrap(np.angle(K1))   # ФЧХ линии связи
    # функция unwrap убирает скачки фазы, когда значение atan превышает |pi|

    plt.figure(1)

    plt.subplot(211) # количество окон по горизонтали, по вертикали, номер окна
    plt.semilogx(f, ACH)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('f, Гц')
    plt.ylabel('|K(f)|')
    plt.title('АЧХ линии связи')


    plt.subplot(212)
    plt.semilogx(f, FCH)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('f, Гц')
    plt.ylabel('angle(f)')
    plt.title('ФЧХ линии связи')

# Задание 2
# ________построение АЧХ и ФЧХ исходного прямоугольного сигнала_______

    A = 1                                           # амплитуда сигнала
    k = f.size                                      # число отсчетов сигнала
    t = np.arange(k)                                # массив от 0 до k
    y1 = A * sg.square(2 * np.pi * t / k, duty=0.5) # Сигнал в диапазоне от -1 до 1

    plt.figure(2)
    plt.subplot(121)
    plt.plot(t, y1, '-b', linewidth=2)
    plt.grid(True)
    # plt.tight_layout(h_pad=1)
    plt.xlabel('N, номер отсчета')
    plt.ylabel('y1(N)')
    plt.title('Исходный сигнал во временной области')

    S1 = np.fft.fft(y1)
    ACH_S1 = np.abs(S1)
    FCH_S1 = np.unwrap(np.angle(S1))

    plt.figure(3)
    plt.subplot(221)
    plt.plot(f, ACH_S1)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('f, Hz')
    plt.ylabel('|K(f)|')
    plt.title('АЧХ исходного сигнала')

    plt.subplot(222)
    plt.plot(f, FCH_S1)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('f, Hz')
    plt.ylabel('angle(f)')
    plt.title('ФЧХ исходного сигнала')


# ------------- построение АЧХ и ФЧХ сигнала после линии связи ------------

    # S1(1)=0       - постоянная сставляющая сигнала
    # S1(2:N/2+1)   - действительная часть спектра
    # S1(N/2+2,N+1) - мнимая часть спектра сигнала
    S1[int(N / 2):] = 0     # зануляем мнимую часть спектра сигнала

    S2 = S1 * K1
    ACH_S2 = np.abs(S2)                 # АЧХ
    FCH_S2 = np.unwrap(np.angle(S2))    # ФЧХ

    plt.subplot(223)
    plt.stem(f, ACH_S2)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('f, Hz')
    plt.ylabel('|K(f)|')
    plt.title('АЧХ принятого сигнала')

    plt.subplot(224)
    plt.plot(f, FCH_S2)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('f, Hz')
    plt.ylabel('angle(f)')
    plt.title('ФЧХ принятого сигнала')

    # ---- восстановление сигнала во временной области после линии связи ------
    # так как ранее половину спектра занулили,
    # то для восстановления начальной амплитуды домножаем сигнал на 2
    y2 = 2 * np.fft.ifft(S2)

    plt.figure(2)
    plt.subplot(122)
    plt.plot(t, np.real(y2), '-r', linewidth=2)
    plt.grid(True)
    plt.tight_layout(h_pad=1)
    plt.xlabel('N, номер отсчета')
    plt.ylabel('y2(N)')
    plt.title('Сигнал после линии связи')

# Задание 3
# ___________________________________________________________

    t = np.arange(0, N) / fs    # временная шкала
    f0 = fsig                   # Частота прямоугольного сигнала

    signal = np.zeros(N)    # Создаем массив для прямоугольного сигнала
    signal[:N // 2] = 1     # Генерируем прямоугольный сигнал

    # Моделируем прохождение через линию связи с затуханием пятой гармоники
    attenuation_factor = 0.5  # Фактор затухания для пятой гармоники
    f5 = 5 * f0  # Частота пятой гармоники
    print('Частота 5-ой гармоники', f5)
    signal_5 = signal.copy()
    signal_5[int(N * f5 / fs):] *= attenuation_factor

    # Выводим графики сигнала и затухшей пятой гармоники
    plt.figure(4)
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Прямоугольный сигнал')
    plt.xlabel('Время, сек')
    plt.ylabel('Амплитуда')

    plt.subplot(2, 1, 2)
    plt.plot(t, signal_5)
    plt.title('Затухшая пятая гармоника')
    plt.xlabel('Время, сек')
    plt.ylabel('Амплитуда')

    plt.tight_layout()


# Задание 4

    # Создание прямоугольного сигнала
    t = np.arange(N)
    y1 = sg.square(2 * np.pi * t / N, duty=0.5)

    # Наложение AWGN на сигнал с разными уровнями энергии шума
    noisy_signal1 = awgn1(y1, SNR=1)
    noisy_signal2 = awgn1(y1, SNR=1 / 2)
    noisy_signal3 = awgn1(y1, SNR=2)

    # Визуализация сигналов
    plt.figure(5)
    plt.subplot(4, 1, 1)
    plt.plot(t, y1, 'b', linewidth=2)
    # plt.axis([0, N - 1, -1.5, 1.5])
    plt.grid(True)
    plt.tight_layout(h_pad=0.05)
    plt.xlabel('Отсчеты')
    plt.ylabel('Амплитуда')
    plt.title('Исходный прямоугольный сигнал')

    plt.subplot(4, 1, 2)
    plt.plot(t, noisy_signal1, 'r', linewidth=2)
    # plt.axis([0, N - 1, -3, 3])
    plt.grid(True)
    plt.tight_layout(h_pad=0.05)
    plt.xlabel('Отсчеты')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал с AWGN (SNR = 1)')

    plt.subplot(4, 1, 3)
    plt.plot(t, noisy_signal2, 'g', linewidth=2)
    # plt.axis([0, N - 1, -3, 3])
    plt.grid(True)
    plt.tight_layout(h_pad=0.05)
    plt.xlabel('Отсчеты')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал с AWGN (SNR = 1/2)')

    plt.subplot(4, 1, 4)
    plt.plot(t, noisy_signal3, 'm', linewidth=2)
    # plt.axis([0, N - 1, -1.5, 1.5])
    plt.grid(True)
    plt.tight_layout(h_pad=0.05)
    # plt.xlabel('Отсчеты')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал с AWGN (SNR = 2)')
    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()