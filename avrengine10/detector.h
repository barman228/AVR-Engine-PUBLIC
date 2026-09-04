#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>

/**
 * @brief Определяет частоту среза (край данных) для каждого аудиоканала.
 * @param pSampleData Указатель на массив сэмплов (interleaved формат)
 * @param channels Количество каналов
 * @param sampleRate Частота дискретизации (Гц)
 * @param totalPCMFrameCount Общее количество фреймов
 * @param threshold_ratio Порог тишины относительно пика (по умолчанию 0.005 или 0.5%)
 * @return std::vector<double> Массив частот среза, где индекс соответствует номеру канала
 */
std::vector<double> detect_cutoff_frequencies(
    const float* pSampleData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    double threshold_ratio = 0.005
);

#endif