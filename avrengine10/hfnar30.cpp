#include "hfnar30.h"
#include "fft.h"
#include "equalizer.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void upscale30_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
) {
    const size_t WINDOW_SIZE_V10 = 512;
    const size_t HOP_SIZE_V10 = WINDOW_SIZE_V10 / 4; // 128

    const size_t WINDOW_SIZE_V20 = 1024;
    const size_t HOP_SIZE_V20 = WINDOW_SIZE_V20 / 4; // 256

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);

    std::vector<double> window512(WINDOW_SIZE_V10);
    for (size_t i = 0; i < WINDOW_SIZE_V10; ++i) {
        window512[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (WINDOW_SIZE_V10 - 1)));
    }

    std::vector<double> window1024(WINDOW_SIZE_V20);
    for (size_t i = 0; i < WINDOW_SIZE_V20; ++i) {
        window1024[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (WINDOW_SIZE_V20 - 1)));
    }

    // Обрабатываем каждый канал независимо
    for (unsigned int ch = 0; ch < channels; ++ch) {
        double cutoff = cutoffs[ch];

        // --- HFNAR 1.0 (Шум) ---
        double src_f_start_v10 = cutoff / 1.66;
        double src_f_end_v10 = cutoff;
        double tgt_f_start_v10 = cutoff;
        double tgt_f_end_v10 = std::min(cutoff * 1.75, 20000.0);

        // --- HFNAR 2.0 (Копирование) ---
        double src_f_start_v20 = cutoff / 1.333;
        double src_f_end_v20 = cutoff / 1.001;
        double tgt_f_start_v20 = cutoff / 1.001;
        double tgt_f_end_v20 = (cutoff - (cutoff / 1.333)) + cutoff;

        // Жесткое ограничение частоты Найквиста или 20 кГц
        double max_possible_freq = std::min(20000.0, (sampleRate / 2.0) - 1.0);
        tgt_f_end_v10 = std::min(tgt_f_end_v10, max_possible_freq);
        tgt_f_end_v20 = std::min(tgt_f_end_v20, max_possible_freq);

        int src_bin_start_v20 = static_cast<int>(src_f_start_v20 * WINDOW_SIZE_V20 / sampleRate);
        int src_bin_end_v20 = static_cast<int>(src_f_end_v20 * WINDOW_SIZE_V20 / sampleRate);
        int tgt_bin_start_v20 = static_cast<int>(tgt_f_start_v20 * WINDOW_SIZE_V20 / sampleRate);
        int tgt_bin_end_v20 = static_cast<int>(tgt_f_end_v20 * WINDOW_SIZE_V20 / sampleRate);
        int cutoff_bin_v20 = static_cast<int>(cutoff * WINDOW_SIZE_V20 / sampleRate);
        int max_bin_v20 = static_cast<int>(WINDOW_SIZE_V20 / 2);

        src_bin_start_v20 = std::max(0, std::min(max_bin_v20, src_bin_start_v20));
        src_bin_end_v20 = std::max(0, std::min(max_bin_v20, src_bin_end_v20));
        tgt_bin_start_v20 = std::max(0, std::min(max_bin_v20, tgt_bin_start_v20));
        tgt_bin_end_v20 = std::max(0, std::min(max_bin_v20, tgt_bin_end_v20));
        cutoff_bin_v20 = std::max(0, std::min(max_bin_v20, cutoff_bin_v20));

        int src_bin_start_v10 = static_cast<int>(src_f_start_v10 * WINDOW_SIZE_V10 / sampleRate);
        int src_bin_end_v10 = static_cast<int>(src_f_end_v10 * WINDOW_SIZE_V10 / sampleRate);
        int tgt_bin_start_v10 = static_cast<int>(tgt_f_start_v10 * WINDOW_SIZE_V10 / sampleRate);
        int tgt_bin_end_v10 = static_cast<int>(tgt_f_end_v10 * WINDOW_SIZE_V10 / sampleRate);
        int max_bin_v10 = static_cast<int>(WINDOW_SIZE_V10 / 2);

        src_bin_start_v10 = std::max(0, std::min(max_bin_v10, src_bin_start_v10));
        src_bin_end_v10 = std::max(0, std::min(max_bin_v10, src_bin_end_v10));
        tgt_bin_start_v10 = std::max(0, std::min(max_bin_v10, tgt_bin_start_v10));
        tgt_bin_end_v10 = std::max(0, std::min(max_bin_v10, tgt_bin_end_v10));


        // ==========================================
        // ПРОХОД 1: Спектральное копирование (Окно 1024)
        // ==========================================
        std::vector<double> out_real_v20(totalPCMFrameCount, 0.0);
        std::vector<double> window_sum_v20(totalPCMFrameCount, 0.0);

        for (size_t i = 0; i < totalPCMFrameCount; i += HOP_SIZE_V20) {
            std::vector<std::complex<double>> fft_buf(WINDOW_SIZE_V20, 0.0);

            // Накладываем окно анализа 1024
            for (size_t j = 0; j < WINDOW_SIZE_V20; ++j) {
                if (i + j < totalPCMFrameCount) {
                    size_t sample_idx = (i + j) * channels + ch;
                    fft_buf[j] = pInputData[sample_idx] * window1024[j];
                }
            }

            fft(fft_buf);

            // Вычисляем энергию донора v2.0
            double src_energy_v20 = 0.0;
            for (int k = src_bin_start_v20; k <= src_bin_end_v20; ++k) {
                src_energy_v20 += std::norm(fft_buf[k]);
            }

            // Принудительно зануляем всё, что выше оригинального среза
            for (int k = cutoff_bin_v20 + 1; k <= max_bin_v20; ++k) {
                fft_buf[k] = 0.0;
            }

            // Выполняем копирование спектральной огибающей
            int tgt_bins_count_v20 = tgt_bin_end_v20 - tgt_bin_start_v20 + 1;
            int src_range_v20 = src_bin_end_v20 - src_bin_start_v20 + 1;

            if (tgt_bins_count_v20 > 0 && src_energy_v20 > 0.0 && src_range_v20 > 0) {
                std::vector<double> temp_mags(tgt_bins_count_v20, 0.0);
                double temp_tgt_energy = 0.0;

                for (int k = tgt_bin_start_v20; k <= tgt_bin_end_v20; ++k) {
                    int src_k = src_bin_start_v20 + (k - tgt_bin_start_v20) % src_range_v20;
                    double mag = std::abs(fft_buf[src_k]);
                    temp_mags[k - tgt_bin_start_v20] = mag;
                    temp_tgt_energy += mag * mag;
                }

                double energy_scale = 1.0;
                if (temp_tgt_energy > 0.0) {
                    energy_scale = std::sqrt(src_energy_v20 / temp_tgt_energy);
                }

                for (int k = tgt_bin_start_v20; k <= tgt_bin_end_v20; ++k) {
                    double final_mag = temp_mags[k - tgt_bin_start_v20] * energy_scale;
                    double phase = dist(rng);

                    fft_buf[k] = std::complex<double>(
                        final_mag * std::cos(phase),
                        final_mag * std::sin(phase)
                    );
                }
            }
			const std::vector<double> copy_gains_db = { 2.0, 1.0, 0.0, 0.0, -1.0 }; // Равные усиления для копирования
            apply_smooth_equalizer(fft_buf, WINDOW_SIZE_V20, sampleRate, tgt_f_start_v20, tgt_f_end_v20, copy_gains_db);

            for (size_t k = 1; k < WINDOW_SIZE_V20 / 2; ++k) {
                fft_buf[WINDOW_SIZE_V20 - k] = std::conj(fft_buf[k]);
            }
            fft_buf[WINDOW_SIZE_V20 / 2] = std::complex<double>(fft_buf[WINDOW_SIZE_V20 / 2].real(), 0.0);

            ifft(fft_buf);

            for (size_t j = 0; j < WINDOW_SIZE_V20; ++j) {
                if (i + j < totalPCMFrameCount) {
                    out_real_v20[i + j] += fft_buf[j].real() * window1024[j];
                    window_sum_v20[i + j] += window1024[j] * window1024[j];
                }
            }
        }


        // ==========================================
        // ПРОХОД 2: Генерация чистого ВЧ шума (Окно 512)
        // ==========================================
        std::vector<double> out_real_v10(totalPCMFrameCount, 0.0);
        std::vector<double> window_sum_v10(totalPCMFrameCount, 0.0);

        for (size_t i = 0; i < totalPCMFrameCount; i += HOP_SIZE_V10) {
            std::vector<std::complex<double>> fft_buf(WINDOW_SIZE_V10, 0.0);

            // Накладываем окно анализа 512
            for (size_t j = 0; j < WINDOW_SIZE_V10; ++j) {
                if (i + j < totalPCMFrameCount) {
                    size_t sample_idx = (i + j) * channels + ch;
                    fft_buf[j] = pInputData[sample_idx] * window512[j];
                }
            }

            fft(fft_buf);

            // Вычисляем энергию донора для шума v1.0
            double src_energy_v10 = 0.0;
            for (int k = src_bin_start_v10; k <= src_bin_end_v10; ++k) {
                src_energy_v10 += std::norm(fft_buf[k]);
            }

            for (int k = 0; k <= max_bin_v10; ++k) {
                fft_buf[k] = 0.0;
            }

            // Генерируем белый шум в целевых бинах v1.0
            int tgt_bins_count_v10 = tgt_bin_end_v10 - tgt_bin_start_v10 + 1;
            if (tgt_bins_count_v10 > 0 && src_energy_v10 > 0.0) {
                double energy_per_bin = src_energy_v10 / tgt_bins_count_v10;
                double target_amplitude = std::sqrt(energy_per_bin);

                for (int k = tgt_bin_start_v10; k <= tgt_bin_end_v10; ++k) {
                    double phase = dist(rng);
                    fft_buf[k] = std::complex<double>(
                        target_amplitude * std::cos(phase),
                        target_amplitude * std::sin(phase)
                    );
                }
            }

			const std::vector<double> noise_gains_db = { -1.0, 0.0, 3.0, -2.0, -4.0 }; // Равные усиления для шума
            apply_smooth_equalizer(fft_buf, WINDOW_SIZE_V10, sampleRate, tgt_f_start_v10, tgt_f_end_v10, noise_gains_db);

            // Восстанавливаем зеркальную сопряженность спектра для IFFT
            for (size_t k = 1; k < WINDOW_SIZE_V10 / 2; ++k) {
                fft_buf[WINDOW_SIZE_V10 - k] = std::conj(fft_buf[k]);
            }
            fft_buf[WINDOW_SIZE_V10 / 2] = std::complex<double>(fft_buf[WINDOW_SIZE_V10 / 2].real(), 0.0);

            ifft(fft_buf);

            // Накладываем окно синтеза 512 и суммируем методом Overlap-Add
            for (size_t j = 0; j < WINDOW_SIZE_V10; ++j) {
                if (i + j < totalPCMFrameCount) {
                    out_real_v10[i + j] += fft_buf[j].real() * window512[j];
                    window_sum_v10[i + j] += window512[j] * window512[j];
                }
            }
        }


        // ==========================================
        // ФИНАЛЬНЫЙ СИНТЕЗ: Суммируем результаты v2.0 и v1.0
        // =========================================
        for (size_t i = 0; i < totalPCMFrameCount; ++i) {
            double sig_v20 = 0.0;
            double sig_v10 = 0.0;

            if (window_sum_v20[i] > 1e-6) {
                sig_v20 = out_real_v20[i] / window_sum_v20[i];
            }

            if (window_sum_v10[i] > 1e-6) {
                sig_v10 = out_real_v10[i] / window_sum_v10[i];
            }

            double sample_val = sig_v20 + sig_v10 / 1.66;

            if (sample_val > 1.0) sample_val = 1.0;
            if (sample_val < -1.0) sample_val = -1.0;

            // Пишем в финальный interleaved-массив
            pOutputData[i * channels + ch] = static_cast<float>(sample_val);
        }
    }
}