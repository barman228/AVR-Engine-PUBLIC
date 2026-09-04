#include "hfnar40a.h"
#include "hfnar40mod.h"
#include "fft.h"
#include "equalizer.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double calculate_hill_factor_local(double cutoff) {
    using namespace HFNAR40A;
    if (cutoff <= FREQ_MIN_LEFT || cutoff >= FREQ_MIN_RIGHT) return FACTOR_MIN;
    double t = 0.0;
    if (cutoff <= FREQ_MAX_HILL) {
        double x = (cutoff - FREQ_MIN_LEFT) / (FREQ_MAX_HILL - FREQ_MIN_LEFT);
        t = x * x * (3.0 - 2.0 * x);
    }
    else {
        double x = (FREQ_MIN_RIGHT - cutoff) / (FREQ_MIN_RIGHT - FREQ_MAX_HILL);
        t = x * x * (3.0 - 2.0 * x);
    }
    return FACTOR_MIN + (FACTOR_MAX - FACTOR_MIN) * t;
}

void upscale40a_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
) {
    using namespace HFNAR40A;

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

    double max_possible_freq = std::min(20000.0, (sampleRate / 2.0) - 1.0);

    for (unsigned int ch = 0; ch < channels; ++ch) {
        double cutoff = cutoffs[ch];
        double factor = calculate_hill_factor_local(cutoff);

        // Передаем настроенные коэффициенты фокусировки полосы анализа
        std::vector<double> drum_limits = HFNAR40::calculate_drum_adaptive_limits(
            pInputData, totalPCMFrameCount, channels, ch, sampleRate, cutoff,
            WINDOW_SIZE_V10, HOP_SIZE_V10, DRUM_FOCUS_LOW_RATIO, DRUM_FOCUS_HIGH_RATIO
        );

        double src_f_start_v10 = cutoff / 1.333;
        double src_f_end_v10 = cutoff;
        double tgt_f_start_v10 = cutoff / 1.222;

        double src_f_start_v20 = cutoff / factor;
        double src_f_end_v20 = cutoff / 1.001;
        double tgt_f_start_v20 = cutoff / 1.001;
        double tgt_f_end_v20 = (cutoff - (cutoff / factor)) + cutoff;
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
        int max_bin_v10 = static_cast<int>(WINDOW_SIZE_V10 / 2);

        src_bin_start_v10 = std::max(0, std::min(max_bin_v10, src_bin_start_v10));
        src_bin_end_v10 = std::max(0, std::min(max_bin_v10, src_bin_end_v10));
        tgt_bin_start_v10 = std::max(0, std::min(max_bin_v10, tgt_bin_start_v10));

        // ==========================================
        // ПРОХОД 1: Спектральное копирование (Окно 1024)
        // ==========================================
        std::vector<double> out_real_v20(totalPCMFrameCount, 0.0);
        std::vector<double> window_sum_v20(totalPCMFrameCount, 0.0);

        for (size_t i = 0; i < totalPCMFrameCount; i += HOP_SIZE_V20) {
            std::vector<std::complex<double>> fft_buf(WINDOW_SIZE_V20, 0.0);

            for (size_t j = 0; j < WINDOW_SIZE_V20; ++j) {
                if (i + j < totalPCMFrameCount) {
                    fft_buf[j] = pInputData[(i + j) * channels + ch] * window1024[j];
                }
            }

            fft(fft_buf);

            double src_energy_v20 = 0.0;
            for (int k = src_bin_start_v20; k <= src_bin_end_v20; ++k) {
                src_energy_v20 += std::norm(fft_buf[k]);
            }

            for (int k = cutoff_bin_v20 + 1; k <= max_bin_v20; ++k) {
                fft_buf[k] = 0.0;
            }

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
                    fft_buf[k] = std::complex<double>(final_mag * std::cos(phase), final_mag * std::sin(phase));
                }
            }

            apply_smooth_equalizer(fft_buf, WINDOW_SIZE_V20, sampleRate, tgt_f_start_v20, tgt_f_end_v20, COPY_GAINS_DB);

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
        // ПРОХОД 2: Адаптивная генерация ВЧ шума (Окно 512)
        // ==========================================
        std::vector<double> out_real_v10(totalPCMFrameCount, 0.0);
        std::vector<double> window_sum_v10(totalPCMFrameCount, 0.0);

        size_t noise_frame_idx = 0;
        for (size_t i = 0; i < totalPCMFrameCount; i += HOP_SIZE_V10) {
            std::vector<std::complex<double>> fft_buf(WINDOW_SIZE_V10, 0.0);

            for (size_t j = 0; j < WINDOW_SIZE_V10; ++j) {
                if (i + j < totalPCMFrameCount) {
                    fft_buf[j] = pInputData[(i + j) * channels + ch] * window512[j];
                }
            }

            fft(fft_buf);

            double src_energy_v10 = 0.0;
            double max_bin_energy_v10 = 0.0;
            int donor_bins_count = 0;

            for (int k = src_bin_start_v10; k <= src_bin_end_v10; ++k) {
                double bin_energy = std::norm(fft_buf[k]);
                src_energy_v10 += bin_energy;
                if (bin_energy > max_bin_energy_v10) max_bin_energy_v10 = bin_energy;
                donor_bins_count++;
            }

            double tonality_scaler = 1.0;
            if (donor_bins_count > 0 && src_energy_v10 > 0.0) {
                double mean_energy = src_energy_v10 / donor_bins_count;
                double crest_factor = max_bin_energy_v10 / mean_energy;

                if (crest_factor > NOISE_CREST_MIN) {
                    double range = NOISE_CREST_MAX - NOISE_CREST_MIN;
                    tonality_scaler = 1.0 - ((crest_factor - NOISE_CREST_MIN) / range);
                    if (tonality_scaler < 0.0) tonality_scaler = 0.0;
                }
            }

            for (int k = 0; k <= max_bin_v10; ++k) {
                fft_buf[k] = 0.0;
            }

            // Динамический лимит на основе отфильтрованной от баса энергии
            double dynamic_multiplier = drum_limits[noise_frame_idx++];
            double dynamic_tgt_f_end_v10 = std::min(cutoff * dynamic_multiplier, 20000.0);
            dynamic_tgt_f_end_v10 = std::min(dynamic_tgt_f_end_v10, max_possible_freq);

            int tgt_bin_end_v10 = static_cast<int>(dynamic_tgt_f_end_v10 * WINDOW_SIZE_V10 / sampleRate);
            tgt_bin_end_v10 = std::max(0, std::min(max_bin_v10, tgt_bin_end_v10));

            int tgt_bins_count_v10 = tgt_bin_end_v10 - tgt_bin_start_v10 + 1;

            if (tgt_bins_count_v10 > 0 && src_energy_v10 > 0.0) {
                double energy_per_bin = src_energy_v10 / tgt_bins_count_v10;
                double target_amplitude = std::sqrt(energy_per_bin) * tonality_scaler;

                for (int k = tgt_bin_start_v10; k <= tgt_bin_end_v10; ++k) {
                    double phase = dist(rng);
                    fft_buf[k] = std::complex<double>(target_amplitude * std::cos(phase), target_amplitude * std::sin(phase));
                }
            }

            apply_smooth_equalizer(fft_buf, WINDOW_SIZE_V10, sampleRate, tgt_f_start_v10, dynamic_tgt_f_end_v10, NOISE_GAINS_DB);

            for (size_t k = 1; k < WINDOW_SIZE_V10 / 2; ++k) {
                fft_buf[WINDOW_SIZE_V10 - k] = std::conj(fft_buf[k]);
            }
            fft_buf[WINDOW_SIZE_V10 / 2] = std::complex<double>(fft_buf[WINDOW_SIZE_V10 / 2].real(), 0.0);

            ifft(fft_buf);

            for (size_t j = 0; j < WINDOW_SIZE_V10; ++j) {
                if (i + j < totalPCMFrameCount) {
                    out_real_v10[i + j] += fft_buf[j].real() * window512[j];
                    window_sum_v10[i + j] += window512[j] * window512[j];
                }
            }
        }

        // ==========================================
        // ОБЪЕДИНЕНИЕ СИГНАЛОВ И МИКСИРОВАНИЕ
        // ==========================================
        for (size_t i = 0; i < totalPCMFrameCount; ++i) {
            double sig_v20 = 0.0;
            double sig_v10 = 0.0;

            if (window_sum_v20[i] > 1e-6) sig_v20 = out_real_v20[i] / window_sum_v20[i];
            if (window_sum_v10[i] > 1e-6) sig_v10 = out_real_v10[i] / window_sum_v10[i];

            double sample_val = sig_v20 + sig_v10 * 1.1;

            if (sample_val > 1.0) sample_val = 1.0;
            if (sample_val < -1.0) sample_val = -1.0;

            pOutputData[i * channels + ch] = static_cast<float>(sample_val);
        }
    }
}