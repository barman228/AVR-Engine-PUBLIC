#ifndef EQUALIZER_H
#define EQUALIZER_H

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Применяет гладкий 5-полосный эквалайзер к комплексной полусфере спектра FFT.
 * @param start_freq Начальная частота диапазона регулирования (например, cutoff)
 * @param end_freq Конечная частота диапазона регулирования (например, Nyquist или 20000 Гц)
 * @param gains_db Вектор из 5 значений усиления в децибелах [band0, band1, band2, band3, band4]
 */
inline void apply_smooth_equalizer(
    std::vector<std::complex<double>>& fft_buf,
    size_t window_size,
    double sampleRate,
    double start_freq,
    double end_freq,
    const std::vector<double>& gains_db
) {
    if (gains_db.size() != 5 || start_freq >= end_freq) return;

    // 1. Преобразуем децибелы в линейные множители амплитуды
    double gains[5];
    for (int i = 0; i < 5; ++i) {
        gains[i] = std::pow(10.0, gains_db[i] / 20.0);
    }

    // 2. Рассчитываем 5 опорных частотных узлов внутри целевого ВЧ-диапазона
    double f_nodes[5];
    for (int i = 0; i < 5; ++i) {
        f_nodes[i] = start_freq + i * (end_freq - start_freq) / 4.0;
    }

    size_t half_size = window_size / 2;

    // 3. Сканируем бины положительных частот спектра
    for (size_t k = 0; k <= half_size; ++k) {
        double f = static_cast<double>(k) * sampleRate / window_size;

        // Корректируем только те бины, которые попадают в наш целевой диапазон апскейла
        if (f < start_freq || f > end_freq) {
            continue;
        }

        double current_gain = 1.0;

        if (f <= f_nodes[0]) {
            current_gain = gains[0];
        }
        else if (f >= f_nodes[4]) {
            current_gain = gains[4];
        }
        else {
            // Ищем, между какими двумя узлами находится частота бина
            for (int i = 0; i < 4; ++i) {
                if (f >= f_nodes[i] && f <= f_nodes[i + 1]) {
                    // Нормализованное положение между узлами (от 0.0 до 1.0)
                    double t = (f - f_nodes[i]) / (f_nodes[i + 1] - f_nodes[i]);

                    // Косинусная интерполяция для идеальной гладкости переходов на стыках
                    double t_smooth = (1.0 - std::cos(t * M_PI)) / 2.0;

                    current_gain = gains[i] * (1.0 - t_smooth) + gains[i + 1] * t_smooth;
                    break;
                }
            }
        }

        // Умножаем комплексную амплитуду бина на рассчитанный плавный коэффициент
        fft_buf[k] *= current_gain;
    }
}

#endif