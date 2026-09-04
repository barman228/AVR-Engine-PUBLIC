#ifndef HFNAR20_H
#define HFNAR20_H

#include <vector>

void upscale2_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
);

#endif