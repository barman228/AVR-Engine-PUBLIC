#ifndef HFNAR30_H
#define HFNAR30_H

#include <vector>

/**
 * @brief Выполняет высокочастотный апскейл аудиодорожки (HFNAR v2.1).
 * Комбинирует спектральное копирование огибающей из v2.0 и подмешивание шума из v1.0.
 * @param pInputData Указатель на оригинальные interleaved float-сэмплы
 * @param pOutputData Указатель на выделенный массив для выходных данных
 * @param channels Количество каналов
 * @param sampleRate Частота дискретизации (Гц)
 * @param totalPCMFrameCount Общее количество фреймов
 * @param cutoffs Вектор частотных срезов для каждого канала
 */
void upscale30_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
);

#endif