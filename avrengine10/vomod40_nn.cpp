#include "vomod40_nn.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace VOMOD40 {

    namespace {
        // структура тензора (Channels x Time)
        struct Matrix {
            int channels;
            int length;
            std::vector<float> data;
            Matrix(int c, int l) : channels(c), length(l), data(c* l, 0.0f) {}
        };

        Matrix apply_conv1d(const Matrix& in, const std::string& prefix, const VoiceModelNN& model, int stride = 1, int padding = 0, int dilation = 1) {
            const auto& w = model.weights.at(prefix + ".weight");
            const auto& b = model.weights.at(prefix + ".bias");
            int out_c = w.dims[0];
            int in_c = w.dims[1];
            int k_size = w.dims[2];
            int out_l = (in.length + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;

            Matrix out(out_c, out_l);
            for (int oc = 0; oc < out_c; ++oc) {
                float bias_val = b.data[oc];
                for (int t = 0; t < out_l; ++t) {
                    float sum = bias_val;
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int k = 0; k < k_size; ++k) {
                            int in_t = t * stride - padding + k * dilation;
                            if (in_t >= 0 && in_t < in.length) {
                                float in_val = in.data[ic * in.length + in_t];
                                float weight_val = w.data[oc * (in_c * k_size) + ic * k_size + k];
                                sum += in_val * weight_val;
                            }
                        }
                    }
                    out.data[oc * out_l + t] = sum;
                }
            }
            return out;
        }

        Matrix apply_conv_transpose1d(const Matrix& in, const std::string& prefix, const VoiceModelNN& model, int stride, int padding) {
            const auto& w = model.weights.at(prefix + ".weight");
            const auto& b = model.weights.at(prefix + ".bias");
            int in_c = w.dims[0];
            int out_c = w.dims[1];
            int k_size = w.dims[2];
            int out_l = (in.length - 1) * stride - 2 * padding + k_size;

            Matrix out(out_c, out_l);
            // Инициализация смещением (Bias)
            for (int oc = 0; oc < out_c; ++oc) {
                for (int t = 0; t < out_l; ++t) out.data[oc * out_l + t] = b.data[oc];
            }

            // Транспонированная свертка (Overlap-Add)
            for (int ic = 0; ic < in_c; ++ic) {
                for (int t = 0; t < in.length; ++t) {
                    float in_val = in.data[ic * in.length + t];
                    for (int oc = 0; oc < out_c; ++oc) {
                        for (int k = 0; k < k_size; ++k) {
                            int out_t = t * stride - padding + k;
                            if (out_t >= 0 && out_t < out_l) {
                                float weight_val = w.data[ic * (out_c * k_size) + oc * k_size + k];
                                out.data[oc * out_l + out_t] += in_val * weight_val;
                            }
                        }
                    }
                }
            }
            return out;
        }

        void apply_gelu(Matrix& m) {
            for (auto& val : m.data) {
                val = 0.5f * val * (1.0f + std::erf(val * 0.70710678f));
            }
        }

        Matrix apply_concat(const Matrix& a, const Matrix& b) {
            Matrix out(a.channels + b.channels, a.length);
            std::copy(a.data.begin(), a.data.end(), out.data.begin());
            std::copy(b.data.begin(), b.data.end(), out.data.begin() + a.data.size());
            return out;
        }

        Matrix apply_resblock(const Matrix& in, const std::string& prefix, const VoiceModelNN& model, int dilation) {
            Matrix x = apply_conv1d(in, prefix + ".conv.0", model, 1, dilation, dilation);
            apply_gelu(x);

            x = apply_conv1d(x, prefix + ".conv.3", model, 1, 0, 1);

            apply_gelu(x);

            // Skip Connection
            for (size_t i = 0; i < x.data.size(); ++i) x.data[i] += in.data[i];
            return x;
        }
    }

    bool load_nn_model(const std::string& filename, VoiceModelNN& model) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        file.read(reinterpret_cast<char*>(&model.magic), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&model.version), sizeof(uint32_t));

        // Поддержка версий 40 и 41
        if (model.magic != 0x344E4E56 || model.version < 40) return false;

        file.read(reinterpret_cast<char*>(&model.sample_rate), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&model.window_size), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&model.hop_size), sizeof(uint32_t));

        file.read(reinterpret_cast<char*>(&model.min_cutoff_hz), sizeof(float));
        file.read(reinterpret_cast<char*>(&model.max_cutoff_hz), sizeof(float));

        file.read(reinterpret_cast<char*>(&model.target_max_freq), sizeof(float));
        file.read(reinterpret_cast<char*>(&model.in_bins), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&model.out_bins), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&model.seq_len), sizeof(uint32_t));

        uint32_t num_tensors = 0;
        file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));

        for (uint32_t i = 0; i < num_tensors; ++i) {
            uint32_t name_len = 0;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));

            std::string name(name_len, '\0');
            file.read(&name[0], name_len);

            uint32_t ndim = 0;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

            Tensor t;
            t.dims.resize(ndim);
            size_t total_elements = 1;
            for (uint32_t d = 0; d < ndim; ++d) {
                file.read(reinterpret_cast<char*>(&t.dims[d]), sizeof(uint32_t));
                total_elements *= t.dims[d];
            }

            t.data.resize(total_elements);
            file.read(reinterpret_cast<char*>(t.data.data()), total_elements * sizeof(float));

            model.weights[name] = t;
        }

        model.is_loaded = true;
        std::cout << "[VOMOD40] Загружена модель. Версия: " << model.version
            << " | Bins In: " << model.in_bins
            << " | Bins Out: " << model.out_bins << "\n";
        return true;
    }

    // ИНФЕРЕНС (UNet-1D + TCN)
// ИНФЕРЕНС (UNet-1D + TCN с фазовым контекстом)
    std::vector<float> VoiceModelNN::predict(const std::vector<float>& lf_mags_seq) {
        uint32_t in_features = in_bins * 2; // Магнитуда + фазовая дельта для каждого бина
        if (lf_mags_seq.size() != seq_len * in_features) {
            return std::vector<float>(seq_len * out_bins * 2, 0.0f);
        }

        try {
            // 1. Конвертация 1D-потока в 2D матрицу [Channels x Time]
            Matrix x(in_features, seq_len);
            for (uint32_t t = 0; t < seq_len; ++t) {
                for (uint32_t c = 0; c < in_bins; ++c) {
                    // Первая половина каналов — сжатые магнитуды низких частот
                    float mag = lf_mags_seq[t * in_features + c];
                    x.data[c * seq_len + t] = std::pow(mag, 0.3f);

                    // Вторая половина каналов — фазовые дельты низких частот (контекст для сети)
                    float p_delta = lf_mags_seq[t * in_features + in_bins + c];
                    x.data[(in_bins + c) * seq_len + t] = p_delta;
                }
            }

            // 2. Encoder
            Matrix x1 = apply_conv1d(x, "enc_in", *this, 1, 1, 1);
            apply_gelu(x1);

            Matrix x2 = apply_conv1d(x1, "down1", *this, 2, 1, 1);
            apply_gelu(x2);

            Matrix x3 = apply_conv1d(x2, "down2", *this, 2, 1, 1);
            apply_gelu(x3);

            // 3. Bottleneck (TCN с Dilated Convolutions)
            Matrix feat = apply_resblock(x3, "tcn.0", *this, 1);
            feat = apply_resblock(feat, "tcn.1", *this, 2);
            feat = apply_resblock(feat, "tcn.2", *this, 4);
            feat = apply_resblock(feat, "tcn.3", *this, 8);

            // 4. Decoder
            Matrix u2 = apply_conv_transpose1d(feat, "up2", *this, 2, 1);
            apply_gelu(u2);
            u2 = apply_concat(u2, x2);

            Matrix u1 = apply_conv_transpose1d(u2, "up1", *this, 2, 1);
            apply_gelu(u1);
            u1 = apply_concat(u1, x1);

            // 5. Выходной слой
            Matrix out = apply_conv1d(u1, "out_conv", *this, 1, 1, 1);

            // 6. Конвертация матрицы обратно в чередующийся 1D-поток
            uint32_t out_channels = out_bins * 2;
            std::vector<float> final_out(seq_len * out_channels, 0.0f);
            for (uint32_t t = 0; t < seq_len; ++t) {
                for (uint32_t c = 0; c < out_channels; ++c) {
                    final_out[t * out_channels + c] = out.data[c * seq_len + t];
                }
            }

            return final_out;

        }
        catch (const std::exception& e) {
            std::cerr << "[VOMOD40] ERROR: Ошибка инференса: " << e.what() << "\n";
            return std::vector<float>(seq_len * out_bins * 2, 0.0f);
        }
    }
    // Пошло оно нахуй я че тебе бля
    void apply_generative_synthesis(
        std::vector<std::complex<double>>& fft_buf,
        const std::vector<float>& predicted_hf_mag,
        const std::vector<float>& predicted_phase_delta,
        std::vector<double>& accumulated_phases,
        uint32_t min_cutoff_bin,
        uint32_t user_cutoff_bin,
        uint32_t out_bins
    ) {
        // Проходим по каждому бину, предсказанному нейросетью
        for (uint32_t k = 0; k < out_bins; ++k) {
            uint32_t current_bin = min_cutoff_bin + 1 + k;
            if (current_bin >= fft_buf.size() / 2) break;

            // Нейросеть выдает точную фазовую поправку (residual) поверх фазовой истории
            double neural_residual = static_cast<double>(predicted_phase_delta[k]);

            // Интеграция фазы с ограничением в диапазоне [-PI, PI] (Самый треш момент)
            accumulated_phases[k] += neural_residual;
            accumulated_phases[k] = std::fmod(accumulated_phases[k] + M_PI, 2.0 * M_PI);
            if (accumulated_phases[k] < 0) accumulated_phases[k] += 2.0 * M_PI;
            accumulated_phases[k] -= M_PI;

            // Применяем магнитуду и фазу выше пользовательского среза
            if (current_bin > user_cutoff_bin) {
                double hf_mag = static_cast<double>(std::max(0.0f, predicted_hf_mag[k]));
                fft_buf[current_bin] = std::polar(hf_mag, accumulated_phases[k]);
            }
        }

        // Зануление мусора выше целевой частоты генерации
        uint32_t max_gen_bin = min_cutoff_bin + out_bins;
        for (uint32_t k = max_gen_bin + 1; k <= fft_buf.size() / 2; ++k) {
            fft_buf[k] = std::complex<double>(0.0, 0.0);
        }
    }
}