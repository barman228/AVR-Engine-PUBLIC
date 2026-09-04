#include <string>

#ifndef HFNAR40_H
#define HFNAR40_H

namespace HFNAR40 {
    void upscale40_audio(
        const float* pInputData,
        float* pOutputData,
        unsigned int channels,
        unsigned int sampleRate,
        unsigned long long totalPCMFrameCount,
        const std::string& modelFilename,
        float user_cutoff_hz
    );
}
#endif