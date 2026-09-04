#pragma once
#include <vector>
#include <string>
#include <complex>
#include <map>
#include <cstdint>

namespace VOMOD40 {
	// Структура тензора для хранения весов нейросети
    struct Tensor {
        std::vector<uint32_t> dims;
        std::vector<float> data;
    };

    class VoiceModelNN {
    public:
        bool is_loaded = false;

        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t sample_rate = 0;
        uint32_t window_size = 0;
        uint32_t hop_size = 0;
        float min_cutoff_hz = 0.0f; // Диапазон среза (нижняя граница)
        float max_cutoff_hz = 0.0f; // Диапазон среза (верхняя граница)
        float target_max_freq = 0.0f;
        uint32_t in_bins = 0;
        uint32_t out_bins = 0;

        std::map<std::string, Tensor> weights;
        uint32_t seq_len = 32;

        std::vector<float> predict(const std::vector<float>& lf_mags_seq);
    };

    bool load_nn_model(const std::string& filename, VoiceModelNN& model);
	// ИНФЕРЕНС (UNet-1D + TCN с фазовым контекстом)
    void apply_generative_synthesis(
        std::vector<std::complex<double>>& fft_buf,
        const std::vector<float>& predicted_hf_mag,
        const std::vector<float>& predicted_phase_delta,
        std::vector<double>& accumulated_phases,
        uint32_t min_cutoff_bin,
        uint32_t user_cutoff_bin,
        uint32_t out_bins
    );
}