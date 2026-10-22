#include "vomod30_nn.h"
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace VOMOD30 {

    // Активация GELU
    inline double gelu(double x) {
        return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
    }

    // Вспомогательная функция развертки фазы wrap_pi в диапазон [-PI, PI]
    inline double wrap_pi(double angle) {
        return angle - 2.0 * M_PI * std::floor((angle + M_PI) / (2.0 * M_PI));
    }

    // Вспомогательный Conv1D слой с прямолинейным расположением памяти [Channels, Sequence_Length]
    static void conv1d_layer(
        const std::vector<double>& in_act,
        std::vector<double>& out_act,
        const std::vector<double>& weights,
        const std::vector<double>& biases,
        uint32_t in_c, uint32_t out_c, uint32_t L, uint32_t K, int pad
    ) {
        out_act.assign(static_cast<size_t>(out_c) * L, 0.0);
        for (uint32_t oc = 0; oc < out_c; ++oc) {
            for (uint32_t t = 0; t < L; ++t) {
                double sum = biases[oc];
                for (uint32_t ic = 0; ic < in_c; ++ic) {
                    for (uint32_t k = 0; k < K; ++k) {
                        int t_in = static_cast<int>(t) + static_cast<int>(k) - pad;
                        if (t_in >= 0 && t_in < static_cast<int>(L)) {
                            size_t w_idx = (static_cast<size_t>(oc) * in_c + ic) * K + k;
                            size_t x_idx = static_cast<size_t>(ic) * L + t_in;
                            sum += weights[w_idx] * in_act[x_idx];
                        }
                    }
                }
                out_act[static_cast<size_t>(oc) * L + t] = sum;
            }
        }
    }

    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Слой GroupNorm (G=16) по временной оси L
    static void group_norm_layer(
        std::vector<double>& x,
        const std::vector<double>& gamma,
        const std::vector<double>& beta,
        uint32_t C, uint32_t L, uint32_t G = 16, double eps = 1e-5
    ) {
        uint32_t C_g = C / G;
        for (uint32_t g = 0; g < G; ++g) {
            double sum = 0.0;
            uint32_t start_c = g * C_g;
            uint32_t end_c = start_c + C_g;
            size_t count = static_cast<size_t>(C_g) * L;

            for (uint32_t c = start_c; c < end_c; ++c) {
                for (uint32_t t = 0; t < L; ++t) {
                    sum += x[c * L + t];
                }
            }
            double mean = sum / count;

            double sq_sum = 0.0;
            for (uint32_t c = start_c; c < end_c; ++c) {
                for (uint32_t t = 0; t < L; ++t) {
                    double diff = x[c * L + t] - mean;
                    sq_sum += diff * diff;
                }
            }
            double var = sq_sum / count;
            double inv_std = 1.0 / std::sqrt(var + eps);

            for (uint32_t c = start_c; c < end_c; ++c) {
                double g_val = gamma[c];
                double b_val = beta[c];
                for (uint32_t t = 0; t < L; ++t) {
                    size_t idx = c * L + t;
                    x[idx] = (x[idx] - mean) * inv_std * g_val + b_val;
                }
            }
        }
    }

    // Слой Squeeze-and-Excitation (SE)
    static void squeeze_excitation_layer(
        std::vector<double>& x,
        const std::vector<double>& fc1_w,
        const std::vector<double>& fc1_b,
        const std::vector<double>& fc2_w,
        const std::vector<double>& fc2_b,
        uint32_t C, uint32_t L
    ) {
        uint32_t red_C = C / 16;

        // 1. Global Average Pooling
        std::vector<double> pooled(C, 0.0);
        for (uint32_t c = 0; c < C; ++c) {
            double sum = 0.0;
            for (uint32_t t = 0; t < L; ++t) {
                sum += x[c * L + t];
            }
            pooled[c] = sum / L;
        }

        // 2. Bottleneck Conv1D 1x1 + GELU
        std::vector<double> s1(red_C, 0.0);
        conv1d_layer(pooled, s1, fc1_w, fc1_b, C, red_C, 1, 1, 0);
        for (double& v : s1) {
            v = gelu(v);
        }

        // 3. Expansion Conv1D 1x1 + Sigmoid
        std::vector<double> scale(C, 0.0);
        conv1d_layer(s1, scale, fc2_w, fc2_b, red_C, C, 1, 1, 0);
        for (double& v : scale) {
            v = sigmoid(v);
        }

        // 4. Канальное взвешивание
        for (uint32_t c = 0; c < C; ++c) {
            double s = scale[c];
            for (uint32_t t = 0; t < L; ++t) {
                x[c * L + t] *= s;
            }
        }
    }

    bool load_nn_model(const std::string& filename, VoiceModelNN& model) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[VOMOD30] ERROR: Failed to open model file " << filename << "\n";
            return false;
        }

        uint32_t magic = 0;
        uint32_t version = 0;

        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != 0x4E4E4F56 || ((version < 31 || (version > 37) && version != 40))) {
            std::cerr << "[VOMOD30] ERROR: Wrong format or version of .vonn file\n";
            return false;
        }
        else if (version == 40) {
            std::cerr << "[VOMOD30] WARNING: Using an old model version (" << version << "). Consider updating to a newer model.\n";
        }

        model.version = version;
        if (version >= 37) {
            file.read(reinterpret_cast<char*>(&model.frames_per_pack), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_input_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_output_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.hidden_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.num_layers), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.max_freq), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.min_read), sizeof(uint32_t));
        }
        else if (version >= 33 && version < 37) {
            file.read(reinterpret_cast<char*>(&model.frames_per_pack), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_input_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_output_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.hidden_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.num_layers), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.max_freq), sizeof(uint32_t));
        }
        else if (version == 32) {
            model.num_layers = 2;
            file.read(reinterpret_cast<char*>(&model.frames_per_pack), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_input_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_output_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.hidden_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.max_freq), sizeof(uint32_t));
        }
        else if (version == 31) {
            model.frames_per_pack = 1;
            model.num_layers = 2;
            file.read(reinterpret_cast<char*>(&model.single_input_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.hidden_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.single_output_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&model.max_freq), sizeof(uint32_t));
        }

        model.input_size = model.single_input_size * model.frames_per_pack;
        model.output_size = model.single_output_size * model.frames_per_pack;

        model.weights.resize(model.num_layers);
        model.biases.resize(model.num_layers);

        for (uint32_t i = 0; i < model.num_layers; ++i) {
            size_t w_size = 0;
            size_t b_size = 0;

            if (model.version >= 36) {
                if (i == 0) {
                    // in_proj (Conv1D 3x1)
                    w_size = static_cast<size_t>(model.single_input_size) * model.hidden_size * 3;
                    b_size = model.hidden_size;
                }
                else if (i == model.num_layers - 1) {
                    // out_proj (Conv1D 1x1)
                    w_size = static_cast<size_t>(model.hidden_size) * model.single_output_size * 1;
                    b_size = model.single_output_size;
                }
                else {
                    // Подоблоки внутри ResConvBlock
                    uint32_t sub_idx = (i - 1) % 6;
                    uint32_t red_size = model.hidden_size / 16;

                    if (sub_idx == 0 || sub_idx == 2) { // conv1 или conv2
                        w_size = static_cast<size_t>(model.hidden_size) * model.hidden_size * 3;
                        b_size = model.hidden_size;
                    }
                    else if (sub_idx == 1 || sub_idx == 3) { // gn1 или gn2
                        w_size = model.hidden_size;
                        b_size = model.hidden_size;
                    }
                    else if (sub_idx == 4) { // se.fc1
                        w_size = static_cast<size_t>(model.hidden_size) * red_size * 1;
                        b_size = red_size;
                    }
                    else if (sub_idx == 5) { // se.fc2
                        w_size = static_cast<size_t>(red_size) * model.hidden_size * 1;
                        b_size = model.hidden_size;
                    }
                }
            }
            else if (model.version == 35) {
                uint32_t in_features = (i == 0) ? model.single_input_size : model.hidden_size;
                uint32_t out_features = (i == model.num_layers - 1) ? model.single_output_size : model.hidden_size;
                uint32_t kernel_size = (i < model.num_layers - 1) ? 3 : 1;
                w_size = static_cast<size_t>(in_features) * out_features * kernel_size;
                b_size = out_features;
            }
            else {
                uint32_t in_features = (i == 0) ? model.input_size : model.hidden_size;
                uint32_t out_features = (i == model.num_layers - 1) ? model.output_size : model.hidden_size;
                w_size = static_cast<size_t>(in_features) * out_features;
                b_size = out_features;
            }

            model.weights[i].resize(w_size);
            model.biases[i].resize(b_size);

            file.read(reinterpret_cast<char*>(model.weights[i].data()), w_size * sizeof(double));
            file.read(reinterpret_cast<char*>(model.biases[i].data()), b_size * sizeof(double));
        }

        if (!file.good()) {
            std::cerr << "[VOMOD30] ERROR: Model file read incomplete\n";
            return false;
        }

        model.is_loaded = true;
        std::cout << "[VOMOD30] Loaded Version: " << model.version << " | Layers: " << model.num_layers << " | SEQ_LEN (FPP): " << model.frames_per_pack << "\n";
        return true;
    }

    std::vector<double> predict_hf_envelope(const std::vector<double>& input_features, const VoiceModelNN& model) {
        if (!model.is_loaded || input_features.size() != model.input_size || model.num_layers == 0) {
            return std::vector<double>(model.output_size, 0.0);
        }

        uint32_t L = model.frames_per_pack;
        std::vector<double> next_activation(model.single_output_size, 0.0);

        if (model.version >= 36) {
            // 1. Входная проекция in_proj + GELU
            std::vector<double> act;
            conv1d_layer(input_features, act, model.weights[0], model.biases[0],
                model.single_input_size, model.hidden_size, L, 3, 1);
            for (double& v : act) {
                v = gelu(v);
            }

            // 2. Выполнение блоков ResConvBlocK
            size_t num_res_blocks = (model.num_layers - 2) / 6;
            for (size_t b = 0; b < num_res_blocks; ++b) {
                uint32_t base = 1 + static_cast<uint32_t>(b * 6);

                uint32_t l_conv1 = base;
                uint32_t l_gn1   = base + 1;
                uint32_t l_conv2 = base + 2;
                uint32_t l_gn2   = base + 3;
                uint32_t l_se1   = base + 4;
                uint32_t l_se2   = base + 5;

                std::vector<double> residual = act;
                std::vector<double> conv1_out;

                // Conv1 -> GroupNorm -> GELU
                conv1d_layer(act, conv1_out, model.weights[l_conv1], model.biases[l_conv1],
                    model.hidden_size, model.hidden_size, L, 3, 1);
                group_norm_layer(conv1_out, model.weights[l_gn1], model.biases[l_gn1],
                    model.hidden_size, L, 16);
                for (double& v : conv1_out) {
                    v = gelu(v);
                }

                // Conv2 -> GroupNorm
                conv1d_layer(conv1_out, act, model.weights[l_conv2], model.biases[l_conv2],
                    model.hidden_size, model.hidden_size, L, 3, 1);
                group_norm_layer(act, model.weights[l_gn2], model.biases[l_gn2],
                    model.hidden_size, L, 16);

                // Squeeze-and-Excitation
                squeeze_excitation_layer(act, model.weights[l_se1], model.biases[l_se1],
                    model.weights[l_se2], model.biases[l_se2],
                    model.hidden_size, L);

                // Residual Connection + GELU
                for (size_t i = 0; i < act.size(); ++i) {
                    act[i] = gelu(act[i] + residual[i]);
                }
            }

            // 3. Выходная линейная проекция out_proj
            uint32_t last_l = model.num_layers - 1;
            conv1d_layer(act, next_activation, model.weights[last_l], model.biases[last_l],
                model.hidden_size, model.single_output_size, L, 1, 0);

            return next_activation;
        }
        else if (model.version == 35) {
            std::vector<double> current_activation = input_features;
            for (uint32_t l = 0; l < model.num_layers; ++l) {
                uint32_t in_c = (l == 0) ? model.single_input_size : model.hidden_size;
                uint32_t out_c = (l == model.num_layers - 1) ? model.single_output_size : model.hidden_size;
                uint32_t K = (l < model.num_layers - 1) ? 3 : 1;
                int pad = (l < model.num_layers - 1) ? 1 : 0;

                conv1d_layer(current_activation, next_activation, model.weights[l], model.biases[l], in_c, out_c, L, K, pad);

                if (l < model.num_layers - 1) {
                    for (double& v : next_activation) {
                        if (v < 0.0) v = 0.0;
                    }
                }
                current_activation = next_activation;
            }
            return current_activation;
        }
        else {
            std::vector<double> current_activation = input_features;
            for (uint32_t l = 0; l < model.num_layers; ++l) {
                uint32_t in_features = (l == 0) ? model.input_size : model.hidden_size;
                uint32_t out_features = (l == model.num_layers - 1) ? model.output_size : model.hidden_size;

                next_activation.assign(out_features, 0.0);

                for (uint32_t i = 0; i < out_features; ++i) {
                    double sum = model.biases[l][i];
                    size_t weight_offset = static_cast<size_t>(i) * in_features;

                    for (uint32_t j = 0; j < in_features; ++j) {
                        sum += model.weights[l][weight_offset + j] * current_activation[j];
                    }

                    next_activation[i] = (l < model.num_layers - 1 && sum < 0.0) ? 0.0 : sum;
                }
                current_activation = next_activation;
            }
            return current_activation;
        }
    }

    std::vector<double> extract_features(
        const std::vector<std::complex<double>>& fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        double cutoff,
        const VoiceModelNN& model,
        std::vector<std::vector<double>>& phase_history
    ) {
        if (!model.is_loaded || model.single_input_size < 6) return {};

        uint32_t min_read_hz = (model.min_read > 0) ? model.min_read : 6000;
        int bin_min_read = static_cast<int>(min_read_hz * window_size / sample_rate);
        int bin_cutoff = static_cast<int>(cutoff * window_size / sample_rate);

        if (bin_cutoff <= bin_min_read + 4) {
            bin_min_read = std::max(4, bin_cutoff - 8);
        }

        int donor_bins = bin_cutoff - bin_min_read + 1;
        std::vector<double> donor_mags(donor_bins, 0.0);
        double donor_sum = 0.0;

        for (int k = 0; k < donor_bins; ++k) {
            donor_mags[k] = std::abs(fft_buf[bin_min_read + k]);
            donor_sum += donor_mags[k];
        }

        // 1. Расчет глобального НЧ контекста
        double lf_sum = 0.0;
        double log_lf_sum = 0.0;
        int lf_count = std::max(1, bin_min_read - 1);
        for (int k = 1; k < bin_min_read; ++k) {
            double m = std::abs(fft_buf[k]);
            lf_sum += m;
            log_lf_sum += std::log(m + 1e-7);
        }
        double lf_mean = (lf_sum / lf_count) + 1e-7;
        double lf_energy_feat = std::clamp(std::log1p(lf_mean * 10.0) / 5.0, 0.0, 2.0);
        double lf_flatness = std::clamp(std::exp(log_lf_sum / lf_count) / lf_mean, 0.0, 1.0);

        double donor_mean = (donor_sum / donor_bins) + 1e-7;
        double tilt_ratio = std::clamp((lf_mean / donor_mean) / 50.0, 0.0, 1.0);

        // 2. HF Onset
        double prev_donor_energy = 0.0;
        if (!phase_history.empty() && !phase_history[0].empty()) {
            prev_donor_energy = phase_history[0][0];
        }
        double hf_diff = std::max(0.0, donor_mean - prev_donor_energy);
        double hf_onset = std::clamp(std::log1p(hf_diff * 20.0), 0.0, 2.0);

        if (phase_history.empty()) phase_history.resize(1);
        phase_history[0] = { donor_mean };

        // 3. Интерполяция параметров (4 побендовые группы: Mag, Texture, Phase Coh, Phase GD)
        size_t num_context_params = 6;
        size_t num_bands = (model.single_input_size - num_context_params) / 4;

        int smooth_kernel = 3;
        int pad_len = smooth_kernel / 2;
        std::vector<double> smooth_donor(donor_bins, 0.0);
        for (int i = 0; i < donor_bins; ++i) {
            double s = 0.0;
            for (int w = -pad_len; w <= pad_len; ++w) {
                int idx = std::clamp(i + w, 0, donor_bins - 1);
                s += donor_mags[idx];
            }
            smooth_donor[i] = std::max(s / smooth_kernel, 1e-7);
        }

        size_t phase_win = std::max<size_t>(2, donor_bins / num_bands);
        int phase_pad = static_cast<int>(phase_win / 2);
        std::vector<double> raw_phase_sharpness(donor_bins, 0.0);
        std::vector<double> raw_group_delay(donor_bins, 0.0);

        for (int i = 0; i < donor_bins; ++i) {
            std::complex<double> complex_sum(0.0, 0.0);
            double abs_sum = 0.0;
            for (int w = -phase_pad; w <= phase_pad; ++w) {
                int idx = std::clamp(i + w, 0, donor_bins - 1);
                std::complex<double> val = fft_buf[bin_min_read + idx];
                complex_sum += val;
                abs_sum += std::abs(val);
            }
            raw_phase_sharpness[i] = std::abs(complex_sum) / (abs_sum + 1e-7);

            if (i > 0) {
                double d_phi = std::arg(fft_buf[bin_min_read + i]) - std::arg(fft_buf[bin_min_read + i - 1]);
                raw_group_delay[i] = wrap_pi(d_phi) / M_PI;
            }
        }

        std::vector<double> features(model.single_input_size, 0.0);

        for (size_t i = 0; i < num_bands; ++i) {
            double norm_pos = static_cast<double>(i) / (num_bands - 1);
            double bin_pos = norm_pos * (donor_bins - 1);
            size_t idx_low = static_cast<size_t>(std::floor(bin_pos));
            size_t idx_high = std::min(idx_low + 1, static_cast<size_t>(donor_bins - 1));
            double weight = bin_pos - idx_low;

            // 1. Mag
            double raw_mag = (1.0 - weight) * donor_mags[idx_low] + weight * donor_mags[idx_high];
            double log_m = std::log1p(raw_mag * 10.0);
            features[i] = std::clamp(log_m / 5.0, 0.0, 2.0);

            // 2. Texture
            double interp_smooth = (1.0 - weight) * smooth_donor[idx_low] + weight * smooth_donor[idx_high];
            double log_s = std::log1p(interp_smooth * 10.0);
            features[num_bands + i] = std::clamp(((log_m + 1e-7) / (log_s + 1e-7)) / 3.0, 0.0, 1.0);

            // 3. Phase Coherence
            double interp_phase = (1.0 - weight) * raw_phase_sharpness[idx_low] + weight * raw_phase_sharpness[idx_high];
            features[num_bands * 2 + i] = std::clamp(interp_phase, 0.0, 1.0);

            // 4. Phase Group Delay
            double interp_gd = (1.0 - weight) * raw_group_delay[idx_low] + weight * raw_group_delay[idx_high];
            features[num_bands * 3 + i] = std::clamp(interp_gd, -1.0, 1.0);
        }

        // 4. Заполнение контекст-блока
        size_t ctx_offset = num_bands * 4;
        features[ctx_offset + 0] = lf_energy_feat;
        features[ctx_offset + 1] = lf_flatness;
        features[ctx_offset + 2] = tilt_ratio;
        features[ctx_offset + 3] = hf_onset;

        double log_donor_sum = 0.0;
        for (int k = 0; k < donor_bins; ++k) log_donor_sum += std::log(donor_mags[k] + 1e-7);
        double donor_flatness = std::clamp(std::exp(log_donor_sum / donor_bins) / donor_mean, 0.0, 1.0);

        features[ctx_offset + 4] = donor_flatness;
        features[ctx_offset + 5] = cutoff / (sample_rate / 2.0);

        return features;
    }

    void apply_patch_synthesis(
        std::vector<std::complex<double>>& fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        double cutoff,
        const std::vector<double>& envelope,
        const VoiceModelNN& model,
        double reconstruct,
        size_t frame_idx
    ) {
        if (envelope.empty()) return;

        uint32_t min_read_hz = (model.min_read > 0) ? model.min_read : 6000;
        int bin_min_read = static_cast<int>(min_read_hz * window_size / sample_rate);
        int bin_cutoff = static_cast<int>(cutoff * window_size / sample_rate);

        if (bin_cutoff <= bin_min_read + 4) {
            bin_min_read = std::max(4, bin_cutoff - 8);
        }

        if (bin_cutoff < 8) return;

        int donor_start = static_cast<int>(bin_cutoff * 0.7);
        double donor_sum = 0.0;
        int donor_bins = std::max(1, bin_cutoff - donor_start);
        for (int k = donor_start + 1; k <= bin_cutoff; ++k) {
            donor_sum += std::abs(fft_buf[k]);
        }
        double donor_energy = (donor_sum / donor_bins) + 1e-6;

        size_t macro_bands = (envelope.size() - 1) / 5;

        int patch_start = donor_start;
        int patch_width = bin_cutoff - patch_start;
        if (patch_width < 4) patch_width = 4;

        int end_bin = static_cast<int>(model.max_freq * window_size / sample_rate);
        end_bin = std::min(end_bin, static_cast<int>(window_size / 2));
        int hf_bins_count = end_bin - bin_cutoff;

        size_t hop_size = window_size / 4;
        const double TWO_PI = 2.0 * M_PI;

        // СИНТЕЗ
        for (int k = bin_cutoff + 1; k <= end_bin; ++k) {
            int offset_raw = k - bin_cutoff - 1;
            int actual_offset = offset_raw % patch_width;
            int tile_idx = offset_raw / patch_width;
            int src_k = patch_start + actual_offset;

            std::complex<double> src_val = fft_buf[src_k];
            double mag_pure = std::abs(src_val);

            // Автоматическое непрерывное отображение индекса бина k на выходной envelope
            double exact_idx = (hf_bins_count > 1) ? static_cast<double>(k - bin_cutoff - 1) / (hf_bins_count - 1) : 0.0;
            exact_idx = std::clamp(exact_idx, 0.0, 1.0) * (macro_bands - 1.0);

            size_t idx1 = static_cast<size_t>(std::floor(exact_idx));
            size_t idx2 = std::min(idx1 + 1, macro_bands - 1);
            double weight = exact_idx - idx1;

            // 1. Декодирование параметров
            double log_gain = (1.0 - weight) * envelope[idx1] + weight * envelope[idx2];
            double raw_ratio = std::expm1(std::max(0.0, log_gain));

double c_freq = (1.0 - weight) * envelope[idx1 + macro_bands] + weight * envelope[idx2 + macro_bands];
            double c_time = (1.0 - weight) * envelope[idx1 + macro_bands * 2] + weight * envelope[idx2 + macro_bands * 2];
            double phase_sharpness = (1.0 - weight) * envelope[idx1 + macro_bands * 3] + weight * envelope[idx2 + macro_bands * 3];
            
            phase_sharpness = std::clamp(phase_sharpness, 0.0, 1.0);

            // интерполяция углов с учетом циклического переноса wrap_pi
            double rot1 = envelope[idx1 + macro_bands * 4] * M_PI;
            double rot2 = envelope[idx2 + macro_bands * 4] * M_PI;
            double diff_rot = wrap_pi(rot2 - rot1);
            double delta_phase_rot = wrap_pi(rot1 + weight * diff_rot);

            // Модуляция амплитуды
            double phi_freq = std::cos(M_PI * static_cast<double>(actual_offset) / static_cast<double>(patch_width));
            double phi_time = std::cos(TWO_PI * static_cast<double>(frame_idx % 8) / 8.0);
            double dct_mod = std::max(0.05, 1.0 + (c_freq * phi_freq * 0.4) + (c_time * phi_time * 0.2));

            double dc_amp = raw_ratio * donor_energy;
            double total_amp = dc_amp * dct_mod;

            double max_allowed = donor_energy * 30.0;
            if (total_amp > max_allowed) {
                double over = total_amp - max_allowed;
                total_amp = max_allowed + (max_allowed * std::log2(1.0 + over / max_allowed));
            }

            // Phase Unwrapping
            if (mag_pure > 1e-9) {
                double delta_k = static_cast<double>(k - src_k);
                double phase_adv_per_frame = TWO_PI * delta_k * static_cast<double>(hop_size) / static_cast<double>(window_size);
                double time_phase_shift = static_cast<double>(frame_idx) * phase_adv_per_frame;

                // убраны сторонние сдвиги tile_group_phase и tile_phase_offset
                double locked_phase = wrap_pi(std::arg(src_val) + time_phase_shift + delta_phase_rot);

                // Статичная по времени фазовая дисперсия
                if (phase_sharpness < 0.99) {
                    double band_noise = std::sin(static_cast<double>(k) * 12.9898);
                    double dispersion_amount = (1.0 - phase_sharpness) * M_PI * 0.5;
                    locked_phase = wrap_pi(locked_phase + band_noise * dispersion_amount);
                }

                fft_buf[k] = std::polar(total_amp, locked_phase);
            }
            else {
                fft_buf[k] = std::complex<double>(total_amp, 0.0);
            }
        }

        for (int k = end_bin + 1; k <= static_cast<int>(window_size / 2); ++k) {
            fft_buf[k] = std::complex<double>(0.0, 0.0);
        }
    }
}