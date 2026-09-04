#ifndef HFNAR40A_H
#define HFNAR40A_H

#include <vector>

namespace HFNAR40A {
    // Оконные параметры
    inline constexpr size_t WINDOW_SIZE_V10 = 512;
    inline constexpr size_t HOP_SIZE_V10 = WINDOW_SIZE_V10 / 4; // 128

    inline constexpr size_t WINDOW_SIZE_V20 = 1024;
    inline constexpr size_t HOP_SIZE_V20 = WINDOW_SIZE_V20 / 4; // 256

    // Фиксированные кривые эквалайзеров
    inline const std::vector<double> COPY_GAINS_DB = { -1.0, -2.0, 1.0, 2.0, 0.0 };
    inline const std::vector<double> NOISE_GAINS_DB = { 2.0, -1.0, 2.0, 3.0, 1.0 };

    // Параметры холма копирования
    inline constexpr double FREQ_MIN_LEFT = 4000.0;
    inline constexpr double FREQ_MAX_HILL = 14000.0;
    inline constexpr double FREQ_MIN_RIGHT = 18000.0;
    inline constexpr double FACTOR_MIN = 1.333;
    inline constexpr double FACTOR_MAX = 1.666;

    // Регулируемая фокусировка детектора ударных (пропорции от cutoff)
    inline constexpr double DRUM_FOCUS_LOW_RATIO = 0.5;   // cutoff / 2
    inline constexpr double DRUM_FOCUS_HIGH_RATIO = 1.0;  // cutoff

    // Параметры подавления тона в шуме
    inline constexpr double NOISE_CREST_MIN = 5.5;
    inline constexpr double NOISE_CREST_MAX = 8.5;
}

void upscale40a_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
);

#endif
