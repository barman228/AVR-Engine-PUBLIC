#ifndef HFNAR40P1_H
#define HFNAR40P1_H

#include <vector>

namespace HFNAR40P {
    inline constexpr size_t WINDOW_SIZE_V10 = 512;
    inline constexpr size_t HOP_SIZE_V10 = WINDOW_SIZE_V10 / 4; // 128

    inline constexpr size_t WINDOW_SIZE_V20 = 1024;
    inline constexpr size_t HOP_SIZE_V20 = WINDOW_SIZE_V20 / 4; // 256

    // Фиксированные кривые эквалайзера
    inline const std::vector<double> COPY_GAINS_DB = { -1.0, -2.0, 1.0, 2.0, 0.0 };
    inline const std::vector<double> NOISE_GAINS_DB = { 2.0, -1.0, 2.0, 3.0, 1.0 };

    // Границы динамического спектрального холма (v4.0)
    inline constexpr double FREQ_MIN_LEFT = 4000.0;    // Левый минимум холма
    inline constexpr double FREQ_MAX_HILL = 14000.0;   // Вершина холма
    inline constexpr double FREQ_MIN_RIGHT = 18000.0;  // Правый минимум холма

    inline constexpr double FACTOR_MIN = 1.333;        // Минимальная ширина копирования
    inline constexpr double FACTOR_MAX = 1.666;        // Максимальная ширина копирования

    // Интеллектуальный детектор тональности для Прохода 2 (Шум)
    // Из-за окна Ханна чистый тон размывается на ~3 бина, поэтому для шума отношение 
    // пикового бина к среднему редко превышает 3.5. Синусоиды выдают значения от 8.0 до 20.0+.
    inline constexpr double NOISE_CREST_MIN = 5.5;     // Ниже этой отметки — чистый шум (генерация 100%)
    inline constexpr double NOISE_CREST_MAX = 8.5;     // Выше этой отметки — чистый тон (генерация 0%)
}

void upscale40p1_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
);

#endif