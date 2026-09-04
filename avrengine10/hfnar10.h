#ifndef HFNAR10_H
#define HFNAR10_H

#include <vector>

void upscale_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
);

#endif