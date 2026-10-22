#include "hfnar40mod.h"
#include "fft.h"
#include <cmath>
#include <algorithm>
#include <complex>

namespace HFNAR40 {

    std::vector<double> calculate_drum_adaptive_limits(
        const float* pInputData,
        unsigned long long totalPCMFrameCount,
        unsigned int channels,
        unsigned int ch,
        unsigned int sampleRate,
        double cutoff,
        size_t window_size,
        size_t hop_size,
        double focus_low_ratio,
        double focus_high_ratio
    ) {
        size_t num_frames = 0;
        for (size_t i = 0; i < totalPCMFrameCount; i += hop_size) {
            num_frames++;
        }

        std::vector<double> limits(num_frames, 1.75);
        if (num_frames == 0) return limits;

        std::vector<double> frame_energy(num_frames, 0.0);

        std::vector<double> analytical_window(window_size);
        for (size_t i = 0; i < window_size; ++i) {
            analytical_window[i] = 0.5 * (1.0 - std::cos(2.0 * 3.14159265358979323846 * i / (window_size - 1)));
        }

        double freq_low = cutoff * focus_low_ratio;
        double freq_high = cutoff * focus_high_ratio;

        int bin_low = static_cast<int>(freq_low * window_size / sampleRate);
        int bin_high = static_cast<int>(freq_high * window_size / sampleRate);
        int max_bin = static_cast<int>(window_size / 2);

        bin_low = std::max(0, std::min(max_bin, bin_low));
        bin_high = std::max(0, std::min(max_bin, bin_high));

        // Измерение кратковременной энергии внутри полосы фокусировки
        size_t f_idx = 0;
        for (size_t i = 0; i < totalPCMFrameCount; i += hop_size) {
            std::vector<std::complex<double>> fft_buf(window_size, 0.0);

            for (size_t j = 0; j < window_size; ++j) {
                if (i + j < totalPCMFrameCount) {
                    fft_buf[j] = pInputData[(i + j) * channels + ch] * analytical_window[j];
                }
            }
            fft(fft_buf);

            // Считаем энергию только в интересующем нас частотном окне -басс отсекается здесь
            double band_energy = 0.0;
            for (int k = bin_low; k <= bin_high; ++k) {
                band_energy += std::norm(fft_buf[k]);
            }

            frame_energy[f_idx++] = band_energy;
        }

        // Ballistics
        double smoothed_history_energy = 0.0;
        double current_multiplier = 1.75;

        const double decay_factor = 0.88; // Скорость угасания
        const int history_lookback = 12;  // Окно усреднения фонового шума полосы

        for (size_t f = 0; f < num_frames; ++f) {
            double e_curr = frame_energy[f];

            double history_sum = 0.0;
            int history_count = 0;
            int start_history = std::max(0, static_cast<int>(f) - history_lookback);

            for (int h = start_history; h < static_cast<int>(f); ++h) {
                history_sum += frame_energy[h];
                history_count++;
            }

            if (history_count > 0) {
                smoothed_history_energy = history_sum / history_count;
            }
            else {
                smoothed_history_energy = e_curr;
            }

            double target_multiplier = 1.75;

            if (smoothed_history_energy > 1e-7) {
                double energy_ratio = e_curr / smoothed_history_energy;

                // Так как полоса стала уже, отношение сигнала к фону (сигнал/шум транзиента) становится чище
                if (energy_ratio > 2.2) {
                    double drum_strength = (energy_ratio - 2.2) / 5.5;
                    drum_strength = std::max(0.0, std::min(1.0, drum_strength));

                    target_multiplier = 1.75 + drum_strength * (2.33 - 1.75);
                }
            }

            if (target_multiplier > current_multiplier) {
                current_multiplier = target_multiplier; // Мгновенный взлет полосы генерации
            }
            else {
                current_multiplier = current_multiplier * decay_factor + 1.75 * (1.0 - decay_factor); // Плавный спад
            }

            limits[f] = current_multiplier;
        }

        return limits;
    }
}