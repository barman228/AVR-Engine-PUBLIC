#include "detector.h"
#include "fft.h"
#include <cmath>

std::vector<double> detect_cutoff_frequencies(
    const float* pSampleData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    double threshold_ratio
) {
    std::vector<double> cutoff_frequencies(channels, 0.0);
    const size_t WINDOW_SIZE = 4096;

    for (unsigned int ch = 0; ch < channels; ++ch) {
        std::vector<double> avg_spectrum(WINDOW_SIZE / 2, 0.0);
        size_t chunks_counted = 0;

        // Накопление спектра окнами по всей длине трека
        for (size_t i = 0; i < totalPCMFrameCount; i += WINDOW_SIZE) {
            std::vector<std::complex<double>> fft_buf(WINDOW_SIZE, 0.0);

            for (size_t j = 0; j < WINDOW_SIZE; ++j) {
                size_t sample_idx = (i + j) * channels + ch;
                if (i + j < totalPCMFrameCount) {
                    fft_buf[j] = pSampleData[sample_idx];
                }
                else {
                    fft_buf[j] = 0.0;
                }
            }

            fft(fft_buf);

            for (size_t j = 0; j < WINDOW_SIZE / 2; ++j) {
                avg_spectrum[j] += std::abs(fft_buf[j]);
            }
            chunks_counted++;
        }

        // Поиск максимальной амплитуды (пика) в среднем спектре
        double max_magnitude = 0.0;
        for (size_t j = 0; j < WINDOW_SIZE / 2; ++j) {
            avg_spectrum[j] /= chunks_counted;
            if (avg_spectrum[j] > max_magnitude) {
                max_magnitude = avg_spectrum[j];
            }
        }

        // Поиск "края" аудиоданных сверху вниз
        double threshold = max_magnitude * threshold_ratio;
        size_t cutoff_bin = 0;

        for (int j = (int)(WINDOW_SIZE / 2) - 1; j >= 0; --j) {
            if (avg_spectrum[j] > threshold) {
                cutoff_bin = j;
                break;
            }
        }

        // Расчет итоговой частоты в Гц для текущего канала
        double cutoff_freq = (double)cutoff_bin * sampleRate / WINDOW_SIZE;
        cutoff_frequencies[ch] = cutoff_freq;
    }

    return cutoff_frequencies;
}