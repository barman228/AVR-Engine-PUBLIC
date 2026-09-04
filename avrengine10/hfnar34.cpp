#include "mono2stereo_nn.h"
#include "fft.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace HFNAR34 {

    const size_t WINDOW_SIZE = 2048;
    const size_t HOP_SIZE = 512;

    static std::vector<double> get_hann_window(size_t size) {
        std::vector<double> win(size);
        for (size_t i = 0; i < size; ++i) {
            win[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
        }
        return win;
    }

    static bool check_channels_identical(const float* pData, unsigned long long frames) {
        for (unsigned long long i = 0; i < frames; ++i) {
            if (std::abs(pData[i * 2] - pData[i * 2 + 1]) > 1e-6f) return false;
        }
        return true;
    }

    void process_mono2stereo(
        const float* pInputData,
        float* pOutputData,
        unsigned int in_channels,
        unsigned int sampleRate,
        unsigned long long totalPCMFrameCount,
        const std::string& modelFilename,
        bool force_mono_flag
    ) {
        bool is_mono_file = (in_channels == 1);
        bool identical_stereo = (in_channels == 2) && check_channels_identical(pInputData, totalPCMFrameCount);
        bool run_stereo_synthesis = is_mono_file || force_mono_flag || identical_stereo;

        if (!run_stereo_synthesis) {
            std::copy(pInputData, pInputData + (totalPCMFrameCount * in_channels), pOutputData);
            return;
        }

        auto nn_model = std::make_unique<MONO2STEREO::StereoModelNN>();
        if (!MONO2STEREO::load_nn_model(modelFilename, *nn_model)) {
            std::cerr << "[MONO2STEREO] ERROR: Can't load Stereo Model.\n";
            std::copy(pInputData, pInputData + (totalPCMFrameCount * in_channels), pOutputData);
            return;
        }

        std::vector<double> window = get_hann_window(WINDOW_SIZE);
        if (totalPCMFrameCount < WINDOW_SIZE) return;

        size_t num_frames = (totalPCMFrameCount - WINDOW_SIZE) / HOP_SIZE + 1;
        const unsigned int OUT_CHANNELS = 2;
        std::vector<double> out_real(totalPCMFrameCount * OUT_CHANNELS, 0.0);
        std::vector<double> window_sum(totalPCMFrameCount, 0.0);

        // ВЫДЕЛЕНИЕ БУФЕРА ВРЕМЕННОГО КОНТЕКСТА
        // Автоматически аллоцирует нужный размер (130 * CONTEXT_FRAMES) на основе заголовка модели
        const size_t SINGLE_FRAME_SIZE = 130;
        std::vector<double> rolling_context(nn_model->input_size, 0.0);

        for (size_t f = 0; f < num_frames; ++f) {
            size_t offset = f * HOP_SIZE;
            std::vector<std::complex<double>> mid_fft_buf(WINDOW_SIZE, 0.0);

            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                double mid_sample = 0.0;
                if (in_channels == 1) {
                    mid_sample = pInputData[offset + j];
                }
                else {
                    mid_sample = (pInputData[(offset + j) * 2] + pInputData[(offset + j) * 2 + 1]) * 0.5;
                }
                mid_fft_buf[j] = mid_sample * window[j];
            }

            fft(mid_fft_buf);

            // 1. Извлекаем признаки только текущего изолированного фрейма (размер 130)
            auto current_features = MONO2STEREO::extract_mid_features(mid_fft_buf, WINDOW_SIZE, sampleRate, *nn_model);

            // 2. Сдвигаем буфер истории влево на 128+2 значений и вставляем новый фрейм в конец
            if (rolling_context.size() >= SINGLE_FRAME_SIZE) {
                std::copy(rolling_context.begin() + SINGLE_FRAME_SIZE, rolling_context.end(), rolling_context.begin());
                std::copy(current_features.begin(), current_features.end(), rolling_context.end() - SINGLE_FRAME_SIZE);
            }

            // 3. Отдаем сети весь склеенный временной стек
            auto side_envelope = MONO2STEREO::predict_side_envelope(rolling_context, *nn_model);

            std::vector<std::complex<double>> left_fft(WINDOW_SIZE, 0.0);
            std::vector<std::complex<double>> right_fft(WINDOW_SIZE, 0.0);

            MONO2STEREO::apply_stereo_synthesis(mid_fft_buf, left_fft, right_fft, WINDOW_SIZE, side_envelope);

            for (size_t k = 1; k < WINDOW_SIZE / 2; ++k) {
                left_fft[WINDOW_SIZE - k] = std::conj(left_fft[k]);
                right_fft[WINDOW_SIZE - k] = std::conj(right_fft[k]);
            }
            left_fft[WINDOW_SIZE / 2] = std::complex<double>(left_fft[WINDOW_SIZE / 2].real(), 0.0);
            right_fft[WINDOW_SIZE / 2] = std::complex<double>(right_fft[WINDOW_SIZE / 2].real(), 0.0);

            ifft(left_fft);
            ifft(right_fft);

            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                if (offset + j < totalPCMFrameCount) {
                    size_t target_L = (offset + j) * OUT_CHANNELS + 0;
                    size_t target_R = (offset + j) * OUT_CHANNELS + 1;

                    out_real[target_L] += left_fft[j].real() * window[j];
                    out_real[target_R] += right_fft[j].real() * window[j];

                    window_sum[offset + j] += window[j] * window[j];
                }
            }
        }

        for (size_t i = 0; i < totalPCMFrameCount; ++i) {
            float sample_L = 0.0f;
            float sample_R = 0.0f;

            if (window_sum[i] > 1e-6) {
                sample_L = static_cast<float>(out_real[i * OUT_CHANNELS + 0] / window_sum[i]);
                sample_R = static_cast<float>(out_real[i * OUT_CHANNELS + 1] / window_sum[i]);
            }

            pOutputData[i * OUT_CHANNELS + 0] = std::clamp(sample_L, -1.0f, 1.0f);
            pOutputData[i * OUT_CHANNELS + 1] = std::clamp(sample_R, -1.0f, 1.0f);
        }
    }
}