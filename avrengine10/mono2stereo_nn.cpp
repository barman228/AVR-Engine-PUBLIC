#include "mono2stereo_nn.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MONO2STEREO {

    std::vector<double> extract_mid_features(
        const std::vector<std::complex<double>>& mid_fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        const StereoModelNN& model
    ) {
        if (!model.is_loaded) return {};

        const size_t SINGLE_FRAME_SIZE = 130; // Базовый фиксированный размер признаков одного кадра
        int max_bin = window_size / 2;
        std::vector<double> features(SINGLE_FRAME_SIZE, 0.0);

        double sum_mag = 0.0;
        for (int k = 1; k < max_bin; ++k) {
            sum_mag += std::abs(mid_fft_buf[k]);
        }
        if (sum_mag < 1e-7) sum_mag = 1e-7;

        size_t num_spectral_bands = SINGLE_FRAME_SIZE - 2; // 128 полос

        for (size_t i = 0; i < num_spectral_bands; ++i) {
            double norm_pos = static_cast<double>(i) / (num_spectral_bands - 1);
            double bin_pos = norm_pos * (max_bin - 1);
            size_t idx_low = static_cast<size_t>(std::floor(bin_pos));
            size_t idx_high = std::min(idx_low + 1, static_cast<size_t>(max_bin - 1));

            if (idx_low == 0) idx_low = 1;
            if (idx_high == 0) idx_high = 1;

            double weight = bin_pos - std::floor(bin_pos);

            double mag_low = std::abs(mid_fft_buf[idx_low]);
            double mag_high = std::abs(mid_fft_buf[idx_high]);

            double interpolated_mag = (1.0 - weight) * mag_low + weight * mag_high;
            features[i] = interpolated_mag / (sum_mag / max_bin);
        }

        features[num_spectral_bands] = std::log10(sum_mag + 1e-5);
        features[num_spectral_bands + 1] = 0.5;

        return features;
    }

    std::vector<double> predict_side_envelope(
        const std::vector<double>& input_features,
        const StereoModelNN& model
    ) {
        if (!model.is_loaded) return {};

        std::vector<double> hidden(model.hidden_size, 0.0);
        std::vector<double> output(model.output_size, 0.0);

        for (size_t i = 0; i < model.hidden_size; ++i) {
            double sum = model.b1[i];
            size_t row_offset = i * model.input_size;
            for (size_t j = 0; j < model.input_size; ++j) {
                sum += input_features[j] * model.W1[row_offset + j];
            }
            hidden[i] = std::max(0.0, sum);
        }

        for (size_t i = 0; i < model.output_size; ++i) {
            double sum = model.b2[i];
            size_t row_offset = i * model.hidden_size;
            for (size_t j = 0; j < model.hidden_size; ++j) {
                sum += hidden[j] * model.W2[row_offset + j];
            }
            output[i] = std::max(0.0, sum);
        }

        return output;
    }

    bool load_nn_model(const std::string& filename, StereoModelNN& model) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;

        uint32_t magic = 0, version = 0;
        in.read(reinterpret_cast<char*>(&magic), 4);
        in.read(reinterpret_cast<char*>(&version), 4);

        if (magic != 0x53544552) return false;

        uint32_t dims[3];
        in.read(reinterpret_cast<char*>(dims), 12);
        model.input_size = dims[0];   // Примет полное значение (130 * CONTEXT_FRAMES)
        model.hidden_size = dims[1];
        model.output_size = dims[2];

        model.W1.resize(model.hidden_size * model.input_size);
        model.b1.resize(model.hidden_size);
        model.W2.resize(model.output_size * model.hidden_size);
        model.b2.resize(model.output_size);

        in.read(reinterpret_cast<char*>(model.W1.data()), model.W1.size() * sizeof(double));
        in.read(reinterpret_cast<char*>(model.b1.data()), model.b1.size() * sizeof(double));
        in.read(reinterpret_cast<char*>(model.W2.data()), model.W2.size() * sizeof(double));
        in.read(reinterpret_cast<char*>(model.b2.data()), model.b2.size() * sizeof(double));

        model.is_loaded = true;
        return true;
    }

    void apply_stereo_synthesis(
        const std::vector<std::complex<double>>& mid_fft_buf,
        std::vector<std::complex<double>>& left_fft_buf,
        std::vector<std::complex<double>>& right_fft_buf,
        size_t window_size,
        const std::vector<double>& side_envelope
    ) {
        if (side_envelope.empty()) return;

        int max_bin = window_size / 2;
        size_t macro_bands = side_envelope.size() / 2;

        left_fft_buf[0] = mid_fft_buf[0];
        right_fft_buf[0] = mid_fft_buf[0];

        for (int k = 1; k < max_bin; ++k) {
            std::complex<double> mid_val = mid_fft_buf[k];
            double mid_mag = std::abs(mid_val);
            double mid_phase = std::arg(mid_val);

            double exact_idx = (static_cast<double>(k) / (max_bin - 1)) * (macro_bands - 1.0);
            size_t idx1 = static_cast<size_t>(std::floor(exact_idx));
            size_t idx2 = std::min(idx1 + 1, macro_bands - 1);
            double weight = exact_idx - idx1;

            double side_gain = (1.0 - weight) * side_envelope[idx1] + weight * side_envelope[idx2];
            double raw_phase = (1.0 - weight) * side_envelope[idx1 + macro_bands] + weight * side_envelope[idx2 + macro_bands];
            double phase_offset = (raw_phase * 2.0 * M_PI) - M_PI;

            double side_mag = mid_mag * side_gain;
            double side_phase = mid_phase + phase_offset;

            std::complex<double> side_val(side_mag * std::cos(side_phase), side_mag * std::sin(side_phase));

            left_fft_buf[k] = mid_val + side_val;
            right_fft_buf[k] = mid_val - side_val;
        }

        for (size_t k = max_bin; k < window_size; ++k) {
            left_fft_buf[k] = { 0.0, 0.0 };
            right_fft_buf[k] = { 0.0, 0.0 };
        }
    }
}