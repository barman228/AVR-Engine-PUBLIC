#ifndef HFNAR34_H
#define HFNAR34_H

#include <string>

namespace HFNAR34 {

    // Основная функция процессинга
    void process_mono2stereo(
        const float* pInputData,
        float* pOutputData,
        unsigned int in_channels,
        unsigned int sampleRate,
        unsigned long long totalPCMFrameCount,
        const std::string& modelFilename,
        bool force_mono_flag // Флаг -mono
    );

}

#endif