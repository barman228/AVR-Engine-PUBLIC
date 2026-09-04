#ifndef HFNAR32_H
#define HFNAR32_H

#include <vector>
#include <string>

namespace HFNAR32 {

    constexpr size_t WINDOW_SIZE = 2048;
    constexpr size_t HOP_SIZE = 512;

    bool generate_voice_model32(
        const float* pRefData,
        unsigned long long totalFrames,
        unsigned int channels,
        unsigned int sampleRate,
        const std::string& modelFilename
    );

    void upscale32_patch_audio(
        const float* pInputData,
        float* pOutputData,
        unsigned int channels,
        unsigned int sampleRate,
        unsigned long long totalPCMFrameCount,
        const std::vector<double>& cutoffs,
        const std::string& modelFilename,
        bool use_tes = false,
        int num_threads = 0
    );
}

#endif