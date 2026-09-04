#ifndef FFT_H
#define FFT_H

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Прямое FFT
inline void fft(std::vector<std::complex<double>>& a) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len;
        std::complex<double> wlen(std::cos(ang), -std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Обратное FFT (IFFT)
inline void ifft(std::vector<std::complex<double>>& a) {
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::conj(a[i]);
    }
    fft(a);
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::conj(a[i]) / static_cast<double>(n);
    }
}

#endif