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

        if (magic != 0x4E4E4F56 || ((version < 31 || (version > 35) && version != 40) )) {
            std::cerr << "[VOMOD30] ERROR: Wrong format or version of .vonn file\n";
            return false;
		}
		else if (version == 40) {
			std::cerr << "[VOMOD30] WARNING: Using an old model version (" << version << "). Consider updating to a newer model.\n";
		}

        model.version = version;

        if (version >= 33) {
            file.read(reinterpret_cast<char*>(&model.frames_per_pack), sizeof(uint32_t)); // В версии 40 здесь хранится SEQ_LEN
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
            uint32_t in_features, out_features, kernel_size = 1;

            if (model.version >= 35) {
                // Для Conv1D
                in_features = (i == 0) ? model.single_input_size : model.hidden_size;
                out_features = (i == model.num_layers - 1) ? model.single_output_size : model.hidden_size;
                // фикс kernel_size=3 для всех слоев, кроме последнего
                kernel_size = (i < model.num_layers - 1) ? 3 : 1;
            }
            else {
                // Linear
                in_features = (i == 0) ? model.input_size : model.hidden_size;
                out_features = (i == model.num_layers - 1) ? model.output_size : model.hidden_size;
                kernel_size = 1;
            }

            size_t w_size = static_cast<size_t>(in_features) * out_features * kernel_size;
            size_t b_size = out_features;

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

        std::vector<double> current_activation = input_features;
        std::vector<double> next_activation;

        if (model.version >= 35 && model.version != 40) {
            uint32_t L = model.frames_per_pack;

            for (uint32_t l = 0; l < model.num_layers; ++l) {
                uint32_t in_c = (l == 0) ? model.single_input_size : model.hidden_size;
                uint32_t out_c = (l == model.num_layers - 1) ? model.single_output_size : model.hidden_size;

                // kernel_size=3 для скрытых слоев
                uint32_t K = (l < model.num_layers - 1) ? 3 : 1;
                int pad = (l < model.num_layers - 1) ? 1 : 0;

                next_activation.assign(out_c * L, 0.0);

                for (uint32_t oc = 0; oc < out_c; ++oc) {
                    for (uint32_t t = 0; t < L; ++t) {
                        double sum = model.biases[l][oc];

                        for (uint32_t ic = 0; ic < in_c; ++ic) {
                            for (uint32_t k = 0; k < K; ++k) {
                                int t_in = static_cast<int>(t) + k - pad;
                                if (t_in >= 0 && t_in < static_cast<int>(L)) {
                                    size_t w_idx = (oc * in_c + ic) * K + k;
                                    size_t x_idx = ic * L + t_in; // формат [Channels, Time]
                                    sum += model.weights[l][w_idx] * current_activation[x_idx];
                                }
                            }
                        }

                        if (l < model.num_layers - 1 && sum < 0.0) sum = 0.0;
                        next_activation[oc * L + t] = sum;
                    }
                }
                current_activation = next_activation;
            }
        }
        else {
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
        }

        return current_activation;
    }

    std::vector<double> extract_features(
        const std::vector<std::complex<double>>& fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        double cutoff,
        const VoiceModelNN& model
    ) {
        if (!model.is_loaded || model.single_input_size < 6) return {};

        int bin_cutoff = static_cast<int>(cutoff * window_size / sample_rate);
        if (bin_cutoff < 8) bin_cutoff = 8;

        std::vector<double> live_mags(bin_cutoff, 0.0);
        double sum_mag = 0.0;
        for (int k = 1; k <= bin_cutoff; ++k) {
            live_mags[k - 1] = std::abs(fft_buf[k]);
            sum_mag += live_mags[k - 1];
        }
        if (sum_mag < 1e-7) sum_mag = 1e-7;
        double mean_mag = sum_mag / bin_cutoff;

        // Вычисляем локальную сглаженную огибающую для текстуры (Moving Average)
        std::vector<double> smooth_mags(bin_cutoff, 1e-7);
        int radius = 4;
        for (int i = 0; i < bin_cutoff; ++i) {
            double local_sum = 0.0;
            int count = 0;
            for (int w = -radius; w <= radius; ++w) {
                int idx = i + w;
                if (idx >= 0 && idx < bin_cutoff) {
                    local_sum += live_mags[idx];
                    count++;
                }
            }
            smooth_mags[i] = (count > 0) ? (local_sum / count) : 1e-7;
        }

        // фикс выделяем память строго под размер ОДНОГО! кадра
        std::vector<double> features(model.single_input_size, 0.0);
        const size_t num_bands = (model.single_input_size - 4) / 3;

        for (size_t i = 0; i < num_bands; ++i) {
            double norm_pos = static_cast<double>(i) / (num_bands - 1);
            double bin_pos = norm_pos * (bin_cutoff - 1);
            size_t idx_low = static_cast<size_t>(std::floor(bin_pos));
            size_t idx_high = std::min(idx_low + 1, static_cast<size_t>(bin_cutoff - 1));
            double weight = bin_pos - idx_low;

            double interp_mag = (1.0 - weight) * live_mags[idx_low] + weight * live_mags[idx_high];
            features[i] = interp_mag / mean_mag;

            double interp_smooth = (1.0 - weight) * smooth_mags[idx_low] + weight * smooth_mags[idx_high];
            double texture = (interp_mag + 1e-7) / (interp_smooth + 1e-7);
            features[num_bands + i] = std::clamp(texture, 0.0, 5.0) / 5.0;

            auto get_phase_diff = [&](size_t idx) {
                int k = idx + 1;
                double angle_curr = std::arg(fft_buf[k]);
                double angle_prev = std::arg(fft_buf[k > 1 ? k - 1 : k]);
                double diff = angle_curr - angle_prev;
                while (diff > M_PI) diff -= 2.0 * M_PI;
                while (diff < -M_PI) diff += 2.0 * M_PI;
                return diff / M_PI;
                };

            double diff_low = get_phase_diff(idx_low);
            double diff_high = get_phase_diff(idx_high);
            features[num_bands * 2 + i] = (1.0 - weight) * diff_low + weight * diff_high;
        }

        double centroid = 0.0;
        for (int k = 1; k <= bin_cutoff; ++k) centroid += k * live_mags[k - 1];
        centroid /= (sum_mag + 1e-7);

        int hfc_start = static_cast<int>(bin_cutoff * 0.85);
        double hfc = 0.0;
        for (int k = hfc_start; k < bin_cutoff; ++k) hfc += live_mags[k - 1];

        double log_sum = 0.0;
        for (int k = 0; k < bin_cutoff; ++k) log_sum += std::log(live_mags[k] + 1e-7);
        double flatness = std::exp(log_sum / bin_cutoff) / (mean_mag + 1e-7);

        features[num_bands * 3] = centroid / bin_cutoff;
        features[num_bands * 3 + 1] = hfc / (sum_mag + 1e-7);
        features[num_bands * 3 + 2] = flatness;
        features[num_bands * 3 + 3] = cutoff / (sample_rate / 2.0);

        return features;
    }


    void apply_patch_synthesis(
        std::vector<std::complex<double>>& fft_buf,
        size_t window_size,
        unsigned int sample_rate,
        double cutoff,
        const std::vector<double>& envelope,
        const VoiceModelNN& model,
        double reconstruct
    ) {
        if (envelope.empty()) return;

        int bin_cutoff = static_cast<int>(cutoff * window_size / sample_rate);
        if (bin_cutoff < 8) return;

        // --- 1. ДЕТЕКТОР БАЗОВЫХ ПАРАМЕТРОВ ---
        double sum_mag = 0.0;
        for (int k = 1; k <= bin_cutoff; ++k) sum_mag += std::abs(fft_buf[k]);
        double flat_mag = sum_mag / bin_cutoff;

        size_t tail_params = (model.version >= 32) ? 3 : 2;
        size_t macro_bands = (envelope.size() - tail_params) / 4;

        double width_norm = std::clamp(envelope[envelope.size() - tail_params], 0.0, 1.5);
        double shift_norm = std::clamp(envelope[envelope.size() - tail_params + 1], 0.0, 1.0);

        double raw_mode = (model.version >= 32) ? envelope.back() : 0.0;
        double mode_blend = std::clamp(raw_mode + (reconstruct - 0.5), 0.0, 1.0);

        double donor_ratio = (cutoff < 6000.0) ? 0.65 : 0.5;
        int patch_start = static_cast<int>(bin_cutoff * donor_ratio);
        int base_width = bin_cutoff - patch_start;
        if (base_width < 4) base_width = 4;

        double width_mult = 0.6 + (width_norm * 0.8);
        int patch_width = static_cast<int>(base_width * width_mult);
        if (patch_width < 4) patch_width = 4;

        int max_shift = base_width / 4;
        int shift_bins = static_cast<int>((shift_norm - 0.5) * 2.0 * max_shift);

        int clamp_max = bin_cutoff - patch_width - 1;
        if (clamp_max < 1) {
            clamp_max = 1;
            patch_width = std::max(4, bin_cutoff - 2);
        }

        patch_start = std::clamp(patch_start + shift_bins, 1, clamp_max);
        int max_available_width = bin_cutoff - patch_start;
        if (max_available_width < 1) max_available_width = 1;

        int end_bin = static_cast<int>(model.max_freq * window_size / sample_rate);
        end_bin = std::min(end_bin, static_cast<int>(window_size / 2));
        int hf_bins_count = end_bin - bin_cutoff;

        // Огибающая для выбеливания
        std::vector<double> env_mags(window_size, 1e-9);
        int smooth_radius = 7;
        for (int i = 1; i <= bin_cutoff; ++i) {
            double sum = 0.0;
            int count = 0;
            for (int w = -smooth_radius; w <= smooth_radius; ++w) {
                int check_idx = i + w;
                if (check_idx >= 1 && check_idx <= bin_cutoff) {
                    sum += std::abs(fft_buf[check_idx]);
                    count++;
                }
            }
            double avg = (count > 0) ? (sum / count) : 1e-9;
            env_mags[i] = std::max(avg, 1e-9);
        }

		// цикл синтеза
        for (int k = bin_cutoff + 1; k <= end_bin; ++k) {
            int actual_offset = ((k - bin_cutoff - 1) % patch_width) % max_available_width;
            int src_k = patch_start + actual_offset;

            std::complex<double> src_val = fft_buf[src_k];
            double mag_pure = std::abs(src_val);

            double exact_idx = (hf_bins_count > 1) ? static_cast<double>(k - bin_cutoff - 1) / (hf_bins_count - 1) : 0.0;
            exact_idx = std::clamp(exact_idx, 0.0, 1.0);
            double ratio = exact_idx;
            exact_idx *= (macro_bands - 1.0);

            size_t idx1 = static_cast<size_t>(std::floor(exact_idx));
            size_t idx2 = std::min(idx1 + 1, macro_bands - 1);
            double weight = exact_idx - idx1;

            double gain = (1.0 - weight) * envelope[idx1] + weight * envelope[idx2];
            gain = std::max(0.0, gain);

            double structure = (1.0 - weight) * envelope[idx1 + macro_bands] + weight * envelope[idx2 + macro_bands];
            structure = std::clamp(structure, 0.0, 1.0);

            double coherence = (1.0 - weight) * envelope[idx1 + macro_bands * 2] + weight * envelope[idx2 + macro_bands * 2];
            coherence = std::clamp(coherence, 0.0, 1.0);

            double nn_phase = (1.0 - weight) * envelope[idx1 + macro_bands * 3] + weight * envelope[idx2 + macro_bands * 3];
            nn_phase = std::clamp(nn_phase, 0.0, 1.0);

            double gain_mult = 0.5 + reconstruct;
            gain *= gain_mult;

            double cymbal_factor = structure * (1.0 - coherence);
            if (cymbal_factor > 0.15) {
                gain *= (1.0 + cymbal_factor * 0.85);
            }

            double max_allowed_coherence = 1.0 - (ratio * 0.8);
            if (coherence > max_allowed_coherence) coherence = max_allowed_coherence;
            // Синтез
            double env_mag = env_mags[src_k];
            double rel_mag = mag_pure / (env_mag + 1e-9);
            double raw_rel = mag_pure / (flat_mag + 1e-9);

            double sharpened_mode = std::clamp(mode_blend * 1.1, 0.0, 1.0);

            // punch_blend: дает до 75% выбеливания на плотном миксе, 
            // но так как на голосе sharpened_mode ~ 0.1, реальный punch_blend будет ~ 0.07 тоетсь 93% чистой гармонической текстуры
            double punch_blend = sharpened_mode * 0.75;
            double mixed_rel = (rel_mag * (1.0 - punch_blend)) + (raw_rel * punch_blend);

            // Динамическая гамма
            double gamma = 1.0 + (coherence * 0.35);

            double base_mag = std::pow(std::min(mixed_rel, 25.0), gamma);
            double total_amp = base_mag * gain * flat_mag;

            // ИСПРАВЛЕНИЕ: Логарифмический софт-клиппер вместо tanh.
            // сохраняет микродинамику перегруженных участков, не превращая их в ебанутый белый шум.
            double max_allowed = flat_mag * (8.0 + (structure * 10.0) + (cymbal_factor * 15.0));
            if (total_amp > max_allowed) {
                double over = total_amp - max_allowed;
                total_amp = max_allowed + (max_allowed * std::log2(1.0 + over / max_allowed));
            }

            // --- ФАЗА ---
            if (mag_pure > 1e-9) {
                double donor_angle = std::arg(src_val);
                double phase_ratio = static_cast<double>(k) / static_cast<double>(src_k);

                double current_ratio = 1.0 + (phase_ratio - 1.0) * nn_phase;
                double target_angle = donor_angle * current_ratio;
                std::complex<double> final_phase(std::cos(target_angle), std::sin(target_angle));

                double blur_amount = 1.0 - coherence;

                // Если это явный громкий пик (raw_rel > 3.0), запрещаем размытие фазы.
                bool is_strong_peak = (raw_rel > 3.0) || (coherence > 0.6);

                if (blur_amount > 0.15 && structure < 0.4 && !is_strong_peak) {
                    if (blur_amount > 0.7) blur_amount = 0.7;
                    std::complex<double> v_left = fft_buf[src_k > patch_start ? src_k - 1 : src_k];
                    std::complex<double> v_right = fft_buf[src_k < patch_start + patch_width - 1 ? src_k + 1 : src_k];
                    std::complex<double> smoothed_val = v_left + src_val + v_right;

                    if (std::abs(smoothed_val) > 1e-9) {
                        double smooth_angle = std::arg(smoothed_val) * current_ratio;
                        std::complex<double> target_phase(std::cos(smooth_angle), std::sin(smooth_angle));
                        final_phase = (final_phase * (1.0 - blur_amount)) + (target_phase * blur_amount);
                    }
                }

                double final_phase_len = std::abs(final_phase);
                if (final_phase_len > 1e-9) final_phase /= final_phase_len;

                fft_buf[k] = final_phase * total_amp;
            }
            else {
                fft_buf[k] = std::complex<double>(total_amp, 0.0);
            }
        }

        // Зануление вч мусора
        for (int k = end_bin + 1; k <= static_cast<int>(window_size / 2); ++k) {
            fft_buf[k] = std::complex<double>(0.0, 0.0);
        }
    }
}