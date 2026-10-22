#include "hfnar32.h"
#include "fft.h" 
#include "vomod30_nn.h" 
#include "equalizer.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <future>
#include <mutex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace HFNAR32 {

    static std::vector<double> get_hann_window(size_t size) {
        std::vector<double> win(size);
        for (size_t i = 0; i < size; ++i) {
            win[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
        }
        return win;
    }

    bool generate_voice_model32(
        const float* pRefData,
        unsigned long long totalFrames,
        unsigned int channels,
        unsigned int sampleRate,
        const std::string& modelFilename
    ) {
        std::cerr << "[VOMOD30] ERROR: VOMOD30 uses neural train.\n";
        return false;
    }

    void upscale32_patch_audio(
        const float* pInputData,
        float* pOutputData,
        unsigned int channels,
        unsigned int sampleRate,
        unsigned long long totalPCMFrameCount,
        const std::vector<double>& cutoffs,
        const std::string& modelFilename,
        bool use_tes,
        int num_threads
    ) {
        auto nn_model = std::make_unique<VOMOD30::VoiceModelNN>();
        if (!VOMOD30::load_nn_model(modelFilename, *nn_model)) {
            std::cerr << "[HFNAR32] STOPERROR: Can't load VONN Model\n";
            std::copy(pInputData, pInputData + (totalPCMFrameCount * channels), pOutputData);
            return;
        }

        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) hw_threads = 1;

        unsigned int actual_threads = 1;
        if (num_threads <= 0) {
            actual_threads = hw_threads;
            std::cerr << "[HFNAR32] INFO: Using " << actual_threads << " threads (auto-detected)\n";
        }
        else {
            actual_threads = std::min(static_cast<unsigned int>(num_threads), hw_threads);
            std::cerr << "[HFNAR32] INFO: Using " << actual_threads << " threads (user-specified)\n";
        }

        std::vector<double> window = get_hann_window(WINDOW_SIZE);
        if (totalPCMFrameCount < WINDOW_SIZE) return;

        size_t num_frames = (totalPCMFrameCount - WINDOW_SIZE) / HOP_SIZE + 1;

        std::vector<double> out_real(totalPCMFrameCount * channels, 0.0);
        std::vector<double> window_sum(totalPCMFrameCount * channels, 0.0);

        uint32_t pack_size = nn_model->frames_per_pack > 0 ? nn_model->frames_per_pack : 1;
        size_t total_packs = (num_frames + pack_size - 1) / pack_size;

        for (unsigned int ch = 0; ch < channels; ++ch) {
            double current_cutoff = cutoffs[ch];
            int bin_cutoff = static_cast<int>(current_cutoff * WINDOW_SIZE / sampleRate);
            bin_cutoff = std::clamp(bin_cutoff, 8, static_cast<int>(WINDOW_SIZE / 2) - 1);

            unsigned int current_channel_threads = std::min(actual_threads, static_cast<unsigned int>(total_packs));
            if (current_channel_threads == 0) current_channel_threads = 1;

            size_t packs_per_thread = (total_packs + current_channel_threads - 1) / current_channel_threads;

            std::vector<std::future<void>> futures;
            std::mutex ola_mutex;

            for (unsigned int t = 0; t < current_channel_threads; ++t) {
                size_t start_pack = t * packs_per_thread;
                size_t end_pack = std::min(start_pack + packs_per_thread, total_packs);

                if (start_pack >= end_pack) continue;

                futures.push_back(std::async(std::launch::async, [=, &nn_model, &window, &out_real, &window_sum, &ola_mutex]() {
                    double prev_gain = 1.0;
                    std::vector<double> smoothed_hf_envelope(nn_model->single_output_size, 0.0);
                    bool is_first_frame = true;
                    std::vector<std::vector<double>> phase_history;

                    for (size_t pack_idx = start_pack; pack_idx < end_pack; ++pack_idx) {
                        size_t f = pack_idx * pack_size;
                        uint32_t current_pack_count = std::min(static_cast<uint32_t>(pack_size), static_cast<uint32_t>(num_frames - f));

                        std::vector<std::vector<std::complex<double>>> pack_fft(current_pack_count, std::vector<std::complex<double>>(WINDOW_SIZE, 0.0));
                        std::vector<double> combined_features(nn_model->single_input_size * pack_size, 0.0);
                        std::vector<double> pack_sum_mag(current_pack_count, 0.0);
                        std::vector<double> last_valid_features;

                        for (uint32_t p = 0; p < current_pack_count; ++p) {
                            size_t frame_idx = f + p;
                            size_t offset = frame_idx * HOP_SIZE;

                            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                                pack_fft[p][j] = pInputData[(offset + j) * channels + ch] * window[j];
                            }
                            fft(pack_fft[p]);

                            for (int k = 1; k <= bin_cutoff; ++k) {
                                pack_sum_mag[p] += std::abs(pack_fft[p][k]);
                            }

                            auto single_features = VOMOD30::extract_features(
                                pack_fft[p], WINDOW_SIZE, sampleRate, current_cutoff, *nn_model, phase_history
                            );

                            for (size_t c = 0; c < nn_model->single_input_size; ++c) {
                                combined_features[c * pack_size + p] = single_features[c];
                            }
                            last_valid_features = single_features;
                        }

                        if (current_pack_count < pack_size) {
                            for (uint32_t pad = current_pack_count; pad < pack_size; ++pad) {
                                for (size_t c = 0; c < nn_model->single_input_size; ++c) {
                                    combined_features[c * pack_size + pad] = last_valid_features[c];
                                }
                            }
                        }

                        auto pack_envelope = VOMOD30::predict_hf_envelope(combined_features, *nn_model);

                        for (uint32_t p = 0; p < current_pack_count; ++p) {
                            size_t frame_idx = f + p; // Глобальный индекс кадра
                            size_t offset = frame_idx * HOP_SIZE;

                            // Извлекаем огибающую конкретно для кадра p (длиной 513) из плоского вектора pack_envelope
                            std::vector<double> hf_envelope(nn_model->single_output_size, 0.0);
                            for (size_t c = 0; c < nn_model->single_output_size; ++c) {
                                hf_envelope[c] = pack_envelope[c * pack_size + p];
                            }

                            if (pack_sum_mag[p] > 1e-5) {
                                if (is_first_frame) {
                                    smoothed_hf_envelope = hf_envelope;
                                    is_first_frame = false;
                                }
                                else {
                                    // Легкая межкадровая фильтрация для устранения дребезга
                                    for (size_t i = 0; i < hf_envelope.size(); ++i) {
                                        smoothed_hf_envelope[i] = (hf_envelope[i] * 0.7) + (smoothed_hf_envelope[i] * 0.3);
                                    }
                                }

                                // ИСПРАВЛЕНИЕ: передаем frame_idx (глобальный индекс) вместо p
                                VOMOD30::apply_patch_synthesis(
                                    pack_fft[p],
                                    WINDOW_SIZE,
                                    sampleRate,
                                    current_cutoff,
                                    smoothed_hf_envelope,
                                    *nn_model,
                                    0.5,
                                    frame_idx
                                );

                                apply_smooth_equalizer(pack_fft[p], WINDOW_SIZE, sampleRate, current_cutoff, sampleRate / 2, { 3.0, 4.0, 3.0, 2.0, 0.0 });

                                if (use_tes) {
                                    std::vector<std::complex<double>> fft_lf = pack_fft[p];
                                    std::vector<std::complex<double>> fft_hf(WINDOW_SIZE, 0.0);

                                    for (int k = bin_cutoff + 1; k < WINDOW_SIZE / 2; ++k) {
                                        fft_hf[k] = pack_fft[p][k];
                                        fft_lf[k] = 0.0;
                                    }

                                    for (size_t k = 1; k < WINDOW_SIZE / 2; ++k) {
                                        fft_lf[WINDOW_SIZE - k] = std::conj(fft_lf[k]);
                                        fft_hf[WINDOW_SIZE - k] = std::conj(fft_hf[k]);
                                    }
                                    fft_lf[WINDOW_SIZE / 2] = std::complex<double>(fft_lf[WINDOW_SIZE / 2].real(), 0.0);
                                    fft_hf[WINDOW_SIZE / 2] = std::complex<double>(fft_hf[WINDOW_SIZE / 2].real(), 0.0);

                                    ifft(fft_lf);
                                    ifft(fft_hf);

                                    size_t sub_blocks = 8;
                                    size_t block_size = WINDOW_SIZE / sub_blocks;
                                    double total_lf_e = 1e-9;
                                    double total_hf_e = 1e-9;
                                    std::vector<double> lf_e(sub_blocks, 0.0);
                                    std::vector<double> hf_e(sub_blocks, 0.0);

                                    for (size_t b = 0; b < sub_blocks; ++b) {
                                        for (size_t i = 0; i < block_size; ++i) {
                                            double l = fft_lf[b * block_size + i].real();
                                            double h = fft_hf[b * block_size + i].real();
                                            lf_e[b] += l * l;
                                            hf_e[b] += h * h;
                                        }
                                        total_lf_e += lf_e[b];
                                        total_hf_e += hf_e[b];
                                    }

                                    double smoothing_factor = 0.85;
                                    std::vector<double> smoothed_gains(sub_blocks);
                                    double running_gain = prev_gain;

                                    for (size_t b = 0; b < sub_blocks; ++b) {
                                        double target_gain = std::sqrt((total_hf_e * (lf_e[b] / total_lf_e) + 1e-9) / (hf_e[b] + 1e-9));
                                        target_gain = std::clamp(target_gain, 0.35, 2.0);
                                        smoothed_gains[b] = running_gain * (1.0 - smoothing_factor) + target_gain * smoothing_factor;
                                        running_gain = smoothed_gains[b];
                                    }

                                    for (size_t b = 0; b < sub_blocks; ++b) {
                                        double g_start = (b == 0) ? prev_gain : smoothed_gains[b - 1];
                                        double g_end = smoothed_gains[b];

                                        for (size_t i = 0; i < block_size; ++i) {
                                            double w = static_cast<double>(i) / block_size;
                                            double gain = g_start * (1.0 - w) + g_end * w;

                                            double shaped_hf = fft_hf[b * block_size + i].real() * gain;
                                            pack_fft[p][b * block_size + i] = std::complex<double>(fft_lf[b * block_size + i].real() + shaped_hf, 0.0);
                                        }
                                    }
                                    prev_gain = running_gain;
                                }
                                else {
                                    for (size_t k = 1; k < WINDOW_SIZE / 2; ++k) {
                                        pack_fft[p][WINDOW_SIZE - k] = std::conj(pack_fft[p][k]);
                                    }
                                    pack_fft[p][WINDOW_SIZE / 2] = std::complex<double>(pack_fft[p][WINDOW_SIZE / 2].real(), 0.0);
                                    ifft(pack_fft[p]);
                                }
                            }
                            else {
                                for (size_t k = 1; k < WINDOW_SIZE / 2; ++k) {
                                    pack_fft[p][WINDOW_SIZE - k] = std::conj(pack_fft[p][k]);
                                }
                                pack_fft[p][WINDOW_SIZE / 2] = std::complex<double>(pack_fft[p][WINDOW_SIZE / 2].real(), 0.0);
                                ifft(pack_fft[p]);
                            }

                            std::vector<double> temp_out_real(WINDOW_SIZE);
                            std::vector<double> temp_window_sum(WINDOW_SIZE);

                            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                                temp_out_real[j] = pack_fft[p][j].real() * window[j];
                                temp_window_sum[j] = window[j] * window[j];
                            }

                            {
                                std::lock_guard<std::mutex> lock(ola_mutex);
                                for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                                    size_t target_idx = (offset + j) * channels + ch;
                                    if (offset + j < totalPCMFrameCount) {
                                        out_real[target_idx] += temp_out_real[j];
                                        window_sum[target_idx] += temp_window_sum[j];
                                    }
                                }
                            }
                        }
                    }
                    }));
            }

            for (auto& fut : futures) {
                fut.get();
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