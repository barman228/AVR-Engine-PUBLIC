#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace mvmod10 {

    struct Tensor {
        int C = 0, H = 0, W = 0;
        std::vector<float> data;
        Tensor() = default;
        Tensor(int c, int h, int w) : C(c), H(h), W(w), data((size_t)c* h* w, 0.f) {}
        size_t numel() const { return (size_t)C * H * W; }
        float* ch(int c) { return data.data() + (size_t)c * H * W; }
        const float* ch(int c) const { return data.data() + (size_t)c * H * W; }
        float* at(int c, int y, int x) { return data.data() + ((size_t)c * H + y) * W + x; }
        const float* at(int c, int y, int x) const { return data.data() + ((size_t)c * H + y) * W + x; }
    };

    struct Weight {
        std::vector<int>   shape;
        std::vector<float> data;
        size_t numel() const { size_t n = 1; for (int d : shape) n *= (size_t)d; return n; }
    };

    class MVNNModel {
    public:
        // Гиперпараметры из заголовка MVNN
        uint32_t in_ch = 3, feat = 96, tile = 32;
        uint32_t e0_blocks = 1, e1_blocks = 2, e2_blocks = 6, e2_use_se = 1;
        uint32_t max_scale = 2, default_scale = 2, n_scales = 1;
        std::vector<uint32_t> scale_ids;
        std::vector<std::pair<uint32_t, uint32_t>> scale_ch_offsets;

        std::unordered_map<std::string, Weight> W;

        bool load(const std::string& path);

        // lr — входной LR [3, H, W] в [0,1]. prev_lr — опционально для motion.
        // state_in / state_out — перенос temporal-состояния между кадрами (или nullptr).
        // scale — целевой апскейл (должен быть в scale_ids).
        // strength — 1.0 = как обучено; != 1.0 даёт residual blend с билинейкой.
        Tensor forward(const Tensor& lr,
            const Tensor* prev_lr,
            const Tensor* state_in,
            Tensor* state_out,
            int scale, float strength,
            bool clamp_out = true) const;

    private:
        Tensor run_expert(const Tensor& x, const char* prefix, int n_blocks) const;
        void run_expert_into(const Tensor& x, Tensor& out,
            const char* prefix, int n_blocks) const;
    };

    // Утилита: Tensor [3, H, W] fp32 [0,1] → uint8 HWC
    inline void tensor_to_uint8_hwc(const Tensor& t, uint8_t* out) {
        int HW = t.H * t.W;
        for (int c = 0; c < 3; ++c) {
            const float* p = t.ch(c);
            for (int i = 0; i < HW; ++i) {
                float v = p[i] * 255.f + 0.5f;
                if (v < 0) v = 0; if (v > 255) v = 255;
                out[i * 3 + c] = (uint8_t)v;
            }
        }
    }

    // Утилита: uint8 HWC → Tensor [3, H, W] fp32 [0,1]
    inline Tensor uint8_hwc_to_tensor(const uint8_t* in, int H, int W) {
        Tensor t(3, H, W);
        int HW = H * W;
        for (int c = 0; c < 3; ++c) {
            float* p = t.ch(c);
            for (int i = 0; i < HW; ++i) p[i] = in[i * 3 + c] * (1.f / 255.f);
        }
        return t;
    }

}