#include "fft.h" 
#include "vomod40_nn.h" 
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace HFNAR40 {

    static std::vector<double> get_hann_window(size_t size) {
        std::vector<double> win(size);
        for (size_t i = 0; i < size; ++i) {
            win[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
        }
        return win;
    }

    void upscale40_audio(
        const float* pInputData,
        float* pOutputData,
        unsigned int channels,
        unsigned int sampleRate,
        unsigned long long totalPCMFrameCount,
        const std::string& modelFilename,
        float user_cutoff_hz
    ) {
        auto nn_model = std::make_unique<VOMOD40::VoiceModelNN>();
        if (!VOMOD40::load_nn_model(modelFilename, *nn_model)) {
            std::cerr << "[HFNAR40] ERROR: Невозможно загрузить VNN4 модель\n";
            std::copy(pInputData, pInputData + (totalPCMFrameCount * channels), pOutputData);
            return;
        }

        // Проверка лимитов
        if (user_cutoff_hz < nn_model->min_cutoff_hz || user_cutoff_hz > nn_model->max_cutoff_hz) {
            std::cerr << "[HFNAR40] WARNING: Запрошенный срез (" << user_cutoff_hz
                << " Гц) выходит за пределы диапазона модели ("
                << nn_model->min_cutoff_hz << " - " << nn_model->max_cutoff_hz << " Гц).\n"
                << "Результаты синтеза могут быть непредсказуемыми.\n";
        }

        uint32_t window_size = nn_model->window_size;
        uint32_t hop_size = nn_model->hop_size;

        float active_cutoff = std::clamp(user_cutoff_hz, nn_model->min_cutoff_hz, nn_model->max_cutoff_hz);
        uint32_t active_cutoff_bin = static_cast<uint32_t>(active_cutoff * window_size / sampleRate);
        uint32_t min_cutoff_bin = static_cast<uint32_t>(nn_model->min_cutoff_hz * window_size / sampleRate);

        std::vector<double> window = get_hann_window(window_size);
        if (totalPCMFrameCount < window_size) return;

        size_t num_frames = (totalPCMFrameCount - window_size) / hop_size + 1;
        uint32_t seq_len = nn_model->seq_len;
        uint32_t in_features = nn_model->in_bins * 2; // Магнитуда + фазовая дельта

        std::vector<double> out_real(totalPCMFrameCount * channels, 0.0);
        std::vector<double> window_sum(totalPCMFrameCount * channels, 0.0);

        for (unsigned int ch = 0; ch < channels; ++ch) {
            std::vector<double> accumulated_phases(nn_model->out_bins, 0.0);
            std::vector<double> last_phases(nn_model->in_bins + 1, 0.0); // Трекинг фаз для дельт

            for (size_t f = 0; f < num_frames; f += seq_len) {
                uint32_t current_pack_count = std::min(static_cast<uint32_t>(seq_len), static_cast<uint32_t>(num_frames - f));

                std::vector<std::vector<std::complex<double>>> pack_fft(
                    current_pack_count, std::vector<std::complex<double>>(window_size, 0.0)
                );

                // Буферы для временного хранения магнитуд и фазовых дельт пакета
                std::vector<std::vector<float>> pack_mags(current_pack_count, std::vector<float>(nn_model->in_bins, 0.0f));
                std::vector<std::vector<float>> pack_deltas(current_pack_count, std::vector<float>(nn_model->in_bins, 0.0f));

                for (uint32_t p = 0; p < current_pack_count; ++p) {
                    size_t frame_idx = f + p;
                    size_t offset = frame_idx * hop_size;

                    for (size_t j = 0; j < window_size; ++j) {
                        pack_fft[p][j] = pInputData[(offset + j) * channels + ch] * window[j];
                    }
                    fft(pack_fft[p]);

                    // Заполнение магнитуд и расчет фазовых дельт
                    for (uint32_t k = 1; k <= nn_model->in_bins; ++k) {
                        if (k <= active_cutoff_bin) {
                            float mag = static_cast<float>(std::abs(pack_fft[p][k]));
                            pack_mags[p][k - 1] = mag;

                            float phase = static_cast<float>(std::arg(pack_fft[p][k]));
                            float delta = phase - static_cast<float>(last_phases[k]);

                            // Нормализация фазовой дельты в диапазон [-PI, PI]
                            delta = std::fmod(delta + static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI));
                            if (delta < 0.0f) delta += 2.0f * static_cast<float>(M_PI);
                            delta -= static_cast<float>(M_PI);

                            pack_deltas[p][k - 1] = delta;
                            last_phases[k] = phase;
                        }
                        else {
                            pack_mags[p][k - 1] = 0.0f;
                            pack_deltas[p][k - 1] = 0.0f;
                            last_phases[k] = static_cast<float>(std::arg(pack_fft[p][k]));
                        }
                    }
                }

                // Формирование единого двусоставного входного вектора для модели
                std::vector<float> lf_mags_seq;
                lf_mags_seq.reserve(seq_len * in_features);

                for (uint32_t p = 0; p < current_pack_count; ++p) {
                    // 1. Сначала записываем все магнитуды текущего кадра
                    for (uint32_t c = 0; c < nn_model->in_bins; ++c) {
                        lf_mags_seq.push_back(pack_mags[p][c]);
                    }
                    // 2. Потом уже записываем все фазовые дельты текущего кадра
                    for (uint32_t c = 0; c < nn_model->in_bins; ++c) {
                        lf_mags_seq.push_back(pack_deltas[p][c]);
                    }
                }

                // Добиваем нулями до полного размера seq_len, если пак короче
                while (lf_mags_seq.size() < seq_len * in_features) {
                    lf_mags_seq.push_back(0.0f);
                }

                auto hf_predictions = nn_model->predict(lf_mags_seq);

                for (uint32_t p = 0; p < current_pack_count; ++p) {
                    size_t frame_idx = f + p;
                    size_t offset = frame_idx * hop_size;

                    uint32_t output_stride = nn_model->out_bins * 2;
                    size_t ptr_offset = p * output_stride;

                    std::vector<float> pred_hf_mag(
                        hf_predictions.begin() + ptr_offset,
                        hf_predictions.begin() + ptr_offset + nn_model->out_bins
                    );

                    std::vector<float> pred_phase_delta(
                        hf_predictions.begin() + ptr_offset + nn_model->out_bins,
                        hf_predictions.begin() + ptr_offset + (nn_model->out_bins * 2)
                    );

                    VOMOD40::apply_generative_synthesis(
                        pack_fft[p],
                        pred_hf_mag,
                        pred_phase_delta,
                        accumulated_phases,
                        min_cutoff_bin,
                        active_cutoff_bin,
                        nn_model->out_bins
                    );

                    for (size_t k = 1; k < window_size / 2; ++k) {
                        pack_fft[p][window_size - k] = std::conj(pack_fft[p][k]);
                    }
                    pack_fft[p][window_size / 2] = std::complex<double>(pack_fft[p][window_size / 2].real(), 0.0);

                    ifft(pack_fft[p]);

                    for (size_t j = 0; j < window_size; ++j) {
                        size_t target_idx = (offset + j) * channels + ch;
                        if (offset + j < totalPCMFrameCount) {
                            out_real[target_idx] += pack_fft[p][j].real() * window[j];
                            window_sum[target_idx] += window[j] * window[j];
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < totalPCMFrameCount * channels; ++i) {
            float sample = 0.0f;
            if (window_sum[i] > 1e-6) {
                sample = static_cast<float>(out_real[i] / window_sum[i]);
            }
            pOutputData[i] = std::clamp(sample, -1.0f, 1.0f);
        }
    }
}