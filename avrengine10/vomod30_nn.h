#ifndef VOMOD30_NN_H
#define VOMOD30_NN_H

#include <vector>
#include <complex>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace VOMOD30 {

    struct VoiceModelNN {
        uint32_t frames_per_pack = 1;
        uint32_t single_input_size = 0;
        uint32_t single_output_size = 0;

        uint32_t input_size = 0;
        uint32_t hidden_size = 0;
        uint32_t num_layers = 2;   // по умолчанию 2 для старых версий, вообще уже неактуально
        uint32_t output_size = 0;
        uint32_t max_freq = 0;
        uint32_t min_read = 2000; // Поле для чтения из заголовка

        // <--- ЗАМЕНЕНЫ ВЕСА НА ВЕКТОР МАТРИЦ ---
        std::vector<std::vector<double>> weights;
        std::vector<std::vector<double>> biases;

        int version = 0; // 31, 32 или 33

        bool is_loaded = false;
    };

    bool load_nn_model(const std::string& filename, VoiceModelNN& model);
    // Функция для извлечения признаков из спектра FFT
    std::vector<double> extract_features(
        const std::vector<std::complex<double>>& fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        double cutoff,
        const VoiceModelNN& model, // <-- Теперь передаем модель, чтобы знать размер INPUT
        std::vector<std::vector<double>>& phase_history // test 
    );
    // Функция для предсказания огибающей высоких частот на основе извлеченных признаков
    std::vector<double> predict_hf_envelope(
        const std::vector<double>& input_features,
        const VoiceModelNN& model
    );
    // Функция для применения синтеза на основе предсказанной огибающей
    void apply_patch_synthesis(
        std::vector<std::complex<double>>& fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        double cutoff,
        const std::vector<double>& envelope,
        const VoiceModelNN& model,
        double reconstruct = 0.5,
        size_t frame_idx = 0
    );
}

#endif