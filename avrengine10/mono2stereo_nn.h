#ifndef MONO2STEREO_NN_H
#define MONO2STEREO_NN_H

#include <vector>
#include <complex>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace MONO2STEREO {

    struct StereoModelNN {
        bool is_loaded = false;

        uint32_t input_size = 0;
        uint32_t hidden_size = 0;
        uint32_t output_size = 0; // Должно делиться на 2 (Gain и Phase для Side-канала)

        std::vector<double> W1;
        std::vector<double> b1;
        std::vector<double> W2;
        std::vector<double> b2;
    };

    bool load_nn_model(const std::string& filename, StereoModelNN& model);

    // Извлечение фичей из Mid-канала
    std::vector<double> extract_mid_features(
        const std::vector<std::complex<double>>& mid_fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        const StereoModelNN& model
    );

    // Предсказание огибающей Side-канала
    std::vector<double> predict_side_envelope(
        const std::vector<double>& input_features,
        const StereoModelNN& model
    );

    // Генерация Side-канала и микширование в L/R
    void apply_stereo_synthesis(
        const std::vector<std::complex<double>>& mid_fft_buf,
        std::vector<std::complex<double>>& left_fft_buf,
        std::vector<std::complex<double>>& right_fft_buf,
        size_t window_size,
        const std::vector<double>& side_envelope
    );
}

#endif