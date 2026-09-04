#ifndef HFNAR21_H
#define HFNAR21_H

#include <vector>

void upscale21_audio(
    const float* pInputData,
    float* pOutputData,
    unsigned int channels,
    unsigned int sampleRate,
    unsigned long long totalPCMFrameCount,
    const std::vector<double>& cutoffs
);

#endif