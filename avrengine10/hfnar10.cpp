#include "hfnar10.h"
#include "fft.h"
#include <cmath>
#include <random>

void upscale_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
) {
    const size_t WINDOW_SIZE = 512;
    const size_t HOP_SIZE = WINDOW_SIZE / 4;

    // генератор случайных чисел для фазы шума
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);

    // Создаем сглаживающее окно Ханна
    std::vector<double> window(WINDOW_SIZE);
    for (size_t i = 0; i < WINDOW_SIZE; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (WINDOW_SIZE - 1)));
    }

    // Обрабатываем каждый канал независимо
    for (unsigned int ch = 0; ch < channels; ++ch) {
        double cutoff = cutoffs[ch];

        // Диапазоны частот
        double src_f_start = cutoff / 1.66;
        double src_f_end = cutoff;
        double tgt_f_start = cutoff;
        double tgt_f_end = std::min(cutoff * 1.75, 20000.0);

        // Переводим частоты в индексы бинов FFT
        int src_bin_start = static_cast<int>(src_f_start * WINDOW_SIZE / sampleRate);
        int src_bin_end = static_cast<int>(src_f_end * WINDOW_SIZE / sampleRate);
        int tgt_bin_start = static_cast<int>(tgt_f_start * WINDOW_SIZE / sampleRate);
        int tgt_bin_end = static_cast<int>(tgt_f_end * WINDOW_SIZE / sampleRate);

        int max_bin = static_cast<int>(WINDOW_SIZE / 2) - 1;
        src_bin_start = std::max(0, std::min(max_bin, src_bin_start));
        src_bin_end = std::max(0, std::min(max_bin, src_bin_end));
        tgt_bin_start = std::max(0, std::min(max_bin, tgt_bin_start));
        tgt_bin_end = std::max(0, std::min(max_bin, tgt_bin_end));

        // Временные буферы для Overlap-Add де-окнизации
        std::vector<double> out_real(totalPCMFrameCount, 0.0);
        std::vector<double> window_sum(totalPCMFrameCount, 0.0);

        for (size_t i = 0; i < totalPCMFrameCount; i += HOP_SIZE) {
            std::vector<std::complex<double>> fft_buf(WINDOW_SIZE, 0.0);

            // Накладываем окно анализа
            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                size_t sample_idx = (i + j) * channels + ch;
                if (i + j < totalPCMFrameCount) {
                    fft_buf[j] = pInputData[sample_idx] * window[j];
                }
            }

            fft(fft_buf);

            // Считаем суммарную энергию (квадраты модулей амплитуд) в донорской области
            double src_energy = 0.0;
            for (int k = src_bin_start; k <= src_bin_end; ++k) {
                src_energy += std::norm(fft_buf[k]);
            }

            // Генерируем шум в целевой области
            int tgt_bins_count = tgt_bin_end - tgt_bin_start + 1;
            if (tgt_bins_count > 0 && src_energy > 0.0) {
                double energy_per_bin = src_energy / tgt_bins_count;
                double target_amplitude = std::sqrt(energy_per_bin);

                for (int k = tgt_bin_start; k <= tgt_bin_end; ++k) {
                    double phase = dist(rng);
                    fft_buf[k] = std::complex<double>(
                        target_amplitude * std::cos(phase),
                        target_amplitude * std::sin(phase)
                    );
                }
            }

            // Восстанавливаем зеркальную сопряженность спектра для IFFT
            for (size_t k = 1; k < WINDOW_SIZE / 2; ++k) {
                fft_buf[WINDOW_SIZE - k] = std::conj(fft_buf[k]);
            }
            fft_buf[WINDOW_SIZE / 2] = std::complex<double>(fft_buf[WINDOW_SIZE / 2].real(), 0.0);

            ifft(fft_buf);

            // Накладываем окно синтеза и суммируем обратно во временную шкалу
            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                if (i + j < totalPCMFrameCount) {
                    out_real[i + j] += fft_buf[j].real() * window[j];
                    window_sum[i + j] += window[j] * window[j];
                }
            }
        }

        // Возвращаем данные в выходной массив с нормализацией энергии окон
        for (size_t i = 0; i < totalPCMFrameCount; ++i) {
            double sample_val = 0.0;
            if (window_sum[i] > 1e-6) {
                sample_val = out_real[i] / window_sum[i];
            }

            // Защита от клиппинга (Hard Limiter) утипути костыли сука
            if (sample_val > 1.0) sample_val = 1.0;
            if (sample_val < -1.0) sample_val = -1.0;

            pOutputData[i * channels + ch] = static_cast<float>(sample_val);
        }
    }
}