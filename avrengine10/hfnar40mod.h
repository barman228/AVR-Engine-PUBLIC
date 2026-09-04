#ifndef HFNAR40MOD_H
#define HFNAR40MOD_H

#include <vector>

namespace HFNAR40 {
    /**
     * @brief Анализирует спектральную энергию в заданном диапазоне частот на наличие транзиентов.
     * @param pInputData Указатель на оригинальные interleaved сэмплы
     * @param totalPCMFrameCount Общее количество фреймов в аудио
     * @param channels Количество каналов
     * @param ch Индекс текущего обрабатываемого канала
     * @param sampleRate Частота дискретизации (Гц)
     * @param cutoff Частота среза
     * @param window_size Размер окна анализа (должен быть степенью двойки для FFT)
     * @param hop_size Шаг смещения окна
     * @param focus_low_ratio Нижняя граница фокусировки относительно cutoff (по умолчанию 0.5 -> cutoff / 2)
     * @param focus_high_ratio Верхняя граница фокусировки относительно cutoff (по умолчанию 1.0 -> cutoff)
     */
    std::vector<double> calculate_drum_adaptive_limits(
        const float* pInputData,
        unsigned long long totalPCMFrameCount,
        unsigned int channels,
        unsigned int ch,
        unsigned int sampleRate,
        double cutoff,
        size_t window_size,
        size_t hop_size,
        double focus_low_ratio = 0.5,
        double focus_high_ratio = 1.0
    );
}

#endif