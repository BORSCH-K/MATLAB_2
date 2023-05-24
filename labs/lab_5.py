import numpy as np
import matplotlib.pyplot as plt


def noise_analysis(noise, name_noise=''):
    t = np.arange(0, 100)  # время (ось x)

    CS = np.fft.fft(noise)  # спектр
    ACH = abs(CS)  # АЧХ
    FCH = np.angle(CS)  # ФЧХ

    corr = np.correlate(noise, noise, mode='full')

    plt.figure(1)
    plt.plot(t, noise)
    plt.title('Белый шум с ' + name_noise + ' распределением')

    plt.figure(2)
    plt.plot(t, CS)
    plt.title('Спектр шума с ' + name_noise + ' распределением')

    plt.figure(3)
    plt.plot(t, ACH)
    plt.title('АЧХ шума с ' + name_noise + ' распределением')

    plt.figure(4)
    plt.plot(t, FCH)
    plt.title('ФЧХ шума с ' + name_noise + ' распределением')

    plt.figure(5)
    plt.hist(ACH, bins=10)
    plt.title('Гистограмма АЧХ шума с ' + name_noise + ' распределением')

    plt.figure(6)
    plt.hist(FCH, bins=10)
    plt.title('Гистограмма ФЧХ шума с ' + name_noise + ' распределением')

    plt.figure(7)
    plt.plot(corr)
    plt.title('Автокорреляционная функция для шума с ' + name_noise + ' распределением')

    plt.show()


def main():
    size_noise = 100

    # Сигнал с нормальным распределением
    noise = np.random.normal(size=size_noise)
    noise_analysis(noise, 'нормальным')

    # Сигнал с равномерным распределением
    # noise = np.random.uniform(-1, 1, size=size_noise)

    # for i in range(0, 100):
    #     temp = np.random.uniform(-1, 1, size=size_noise)
    #     for j in range(0, noise.size):
    #         noise[j] += temp[j]

    # noise_analysis(noise, 'равномерным')


if __name__ == '__main__':
    main()
