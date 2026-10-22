#include "mvmod10_filters.h"
#include <cmath>
#include <vector>
#include <cstring>

namespace mvmod10 {

    static inline float luma3(float r, float g, float b) {
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    static inline float smoothstep(float x, float lo, float hi) {
        if (x <= lo) return 0.f;
        if (x >= hi) return 1.f;
        float t = (x - lo) / (hi - lo);
        return t * t * (3.f - 2.f * t);
    }

    // -------- 3x3 gaussian blur (single channel) --------
    static void blur3x3(const float* src, float* dst, int H, int W) {
        static const float k[9] = {
            1.f / 16, 2.f / 16, 1.f / 16,
            2.f / 16, 4.f / 16, 2.f / 16,
            1.f / 16, 2.f / 16, 1.f / 16
        };
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float acc = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    int ys = y + ky;
                    if (ys < 0) ys = -ys;
                    if (ys >= H) ys = 2 * H - ys - 2;
                    if (ys < 0) ys = 0; if (ys >= H) ys = H - 1;
                    const float* srow = src + (size_t)ys * W;
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xs = x + kx;
                        if (xs < 0) xs = -xs;
                        if (xs >= W) xs = 2 * W - xs - 2;
                        if (xs < 0) xs = 0; if (xs >= W) xs = W - 1;
                        acc += srow[xs] * k[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                dst[(size_t)y * W + x] = acc;
            }
        }
    }

    // -------- 5x5 gaussian blur (single channel) --------
    static void blur5x5(const float* src, float* dst, int H, int W) {
        static const float k[25] = {
            1,  4,  6,  4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1,  4,  6,  4, 1
        };
        const float inv = 1.f / 256.f;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float acc = 0.f;
                for (int ky = -2; ky <= 2; ++ky) {
                    int ys = y + ky;
                    if (ys < 0) ys = -ys;
                    if (ys >= H) ys = 2 * H - ys - 2;
                    if (ys < 0) ys = 0; if (ys >= H) ys = H - 1;
                    const float* srow = src + (size_t)ys * W;
                    for (int kx = -2; kx <= 2; ++kx) {
                        int xs = x + kx;
                        if (xs < 0) xs = -xs;
                        if (xs >= W) xs = 2 * W - xs - 2;
                        if (xs < 0) xs = 0; if (xs >= W) xs = W - 1;
                        acc += srow[xs] * k[(ky + 2) * 5 + (kx + 2)];
                    }
                }
                dst[(size_t)y * W + x] = acc * inv;
            }
        }
    }

    // -------- Sobel magnitude (single channel) --------
    static void gradient_mag(const float* src, float* dst, int H, int W) {
        static const int SX[9] = { -1,0,1,-2,0,2,-1,0,1 };
        static const int SY[9] = { -1,-2,-1,0,0,0,1,2,1 };
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float gx = 0.f, gy = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    int ys = y + ky; if (ys < 0 || ys >= H) continue;
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xs = x + kx; if (xs < 0 || xs >= W) continue;
                        float v = src[(size_t)ys * W + xs];
                        int idx = (ky + 1) * 3 + (kx + 1);
                        gx += v * SX[idx];
                        gy += v * SY[idx];
                    }
                }
                dst[(size_t)y * W + x] = std::sqrt(gx * gx + gy * gy);
            }
        }
    }

    // =================== DEHALO ===================
//
// Идея (упрощённый DeHalo_alpha):
//   1. luma = 0.299R + 0.587G + 0.114B
//   2. grad = Sobel(luma)              — где есть границы
//   3. blur_grad = blur5x5(grad)       — «зона вокруг границ»
//   4. halo = max(0, blur_grad - grad) — именно ореолы (не сами границы)
//   5. В зоне halo подмешиваем blur(luma) обратно к luma.
//      Тёмные ореолы прижимаем дополнительно (dark_bias).
//   RGB сдвигаем на ту же дельту luma → цвет не съезжает.
//
    static void dehalo_inplace(Tensor& t, float strength, float dark_bias)
    {
        int H = t.H, W = t.W, HW = H * W;

        std::vector<float> lum(HW), blurred(HW), grad(HW), blur_grad(HW);
        for (int i = 0; i < HW; ++i)
            lum[i] = luma3(t.data[i], t.data[HW + i], t.data[2 * HW + i]);

        blur5x5(lum.data(), blurred.data(), H, W);
        gradient_mag(lum.data(), grad.data(), H, W);
        blur5x5(grad.data(), blur_grad.data(), H, W);

        // Нормализация градиента — чтобы «зона ореола» была стабильной при разных
        // масштабах яркости. Ориентируемся на 95-й перцентиль.
        std::vector<float> sorted_grad = grad;
        std::nth_element(sorted_grad.begin(),
            sorted_grad.begin() + (int)(sorted_grad.size() * 0.95),
            sorted_grad.end());
        float gmax = sorted_grad[(size_t)(sorted_grad.size() * 0.95)] + 1e-6f;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < HW; ++i) {
            float gn = grad[i] / gmax;
            float bn = blur_grad[i] / gmax;

            // halo: положительная часть разницы (blur − sharp), только там,
            // где сама граница не слишком сильная (иначе это уже контур).
            float halo = bn - gn;
            if (halo < 0.f) halo = 0.f;

            // edge_gate: подавляем эффект в самых сильных границах,
            // чтобы не размывать сам контур.
            float edge_gate = 1.f - smoothstep(gn, 0.7f, 1.2f);

            float w = strength * halo * edge_gate;
            if (w <= 0.f) continue;
            if (w > 1.f) w = 1.f;

            // Темнее локального среднего? → усиливаем прижатие к blur.
            float d = lum[i] - blurred[i];
            if (d < 0.f) {
                float dark_boost = 1.f + dark_bias * smoothstep(-d, 0.02f, 0.10f);
                w *= dark_boost;
                if (w > 1.f) w = 1.f;
            }

            float new_luma = lum[i] + (blurred[i] - lum[i]) * w;
            float delta = new_luma - lum[i];

            t.data[i] += delta;
            t.data[HW + i] += delta;
            t.data[2 * HW + i] += delta;
        }
    }

    // =================== PRE ===================

    void apply_pre_filter(const Tensor& in, Tensor& out,
        FilterState& st, const FilterOptions& opt)
    {
        int W = in.W, H = in.H;
        int HW = H * W;
        st.ensure(W, H);

        // 1. Temporal denoise
        if (opt.temporal_denoise) {
            if (out.C != in.C || out.H != in.H || out.W != in.W)
                out = Tensor(in.C, in.H, in.W);

            if (!st.initialized) {
                std::memcpy(out.data.data(), in.data.data(),
                    in.data.size() * sizeof(float));
                for (int i = 0; i < HW; ++i)
                    st.history_luma.data[i] = luma3(
                        in.data[i], in.data[HW + i], in.data[2 * HW + i]);
                std::memcpy(st.history_rgb.data.data(),
                    in.data.data(), in.data.size() * sizeof(float));
                std::memcpy(st.prev_luma.data.data(),
                    st.history_luma.data.data(), HW * sizeof(float));
                st.initialized = true;
            }
            else {
                float denoise = opt.denoise_strength;
                float mt = opt.denoise_motion_thresh;
                float mt_hi = mt * 3.f;

#pragma omp parallel for schedule(static)
                for (int i = 0; i < HW; ++i) {
                    float cur_luma = luma3(
                        in.data[i], in.data[HW + i], in.data[2 * HW + i]);
                    float prev_luma_v = st.prev_luma.data[i];
                    float motion = std::fabs(cur_luma - prev_luma_v);

                    float alpha = 1.f - denoise * (1.f - smoothstep(motion, mt, mt_hi));
                    float inv = 1.f - alpha;

                    out.data[i] = alpha * in.data[i] + inv * st.history_rgb.data[i];
                    out.data[HW + i] = alpha * in.data[HW + i] + inv * st.history_rgb.data[HW + i];
                    out.data[2 * HW + i] = alpha * in.data[2 * HW + i] + inv * st.history_rgb.data[2 * HW + i];
                }

                std::memcpy(st.history_rgb.data.data(), out.data.data(),
                    out.data.size() * sizeof(float));
                for (int i = 0; i < HW; ++i)
                    st.history_luma.data[i] = luma3(
                        out.data[i], out.data[HW + i], out.data[2 * HW + i]);
                // prev_luma = СЫРОЙ вход текущего кадра (для корректного motion на след. кадре)
                for (int i = 0; i < HW; ++i)
                    st.prev_luma.data[i] = luma3(
                        in.data[i], in.data[HW + i], in.data[2 * HW + i]);
            }
        }
        else {
            out = in;
        }

        // 2. Pre-sharp (мелкая текстура, не трогает сильные грани)
        if (opt.pre_sharp && opt.pre_sharp_amount > 0.f) {
            std::vector<float> lum(HW), blur(HW), grad(HW);
            for (int i = 0; i < HW; ++i)
                lum[i] = luma3(out.data[i], out.data[HW + i], out.data[2 * HW + i]);
            blur3x3(lum.data(), blur.data(), H, W);
            gradient_mag(lum.data(), grad.data(), H, W);

            const float amt = opt.pre_sharp_amount;
            const float edge_lo = 0.06f, edge_hi = 0.18f;

#pragma omp parallel for schedule(static)
            for (int i = 0; i < HW; ++i) {
                float detail = lum[i] - blur[i];
                float gate = 1.f - smoothstep(grad[i], edge_lo, edge_hi);
                float d = amt * gate * detail;
                out.data[i] += d;
                out.data[HW + i] += d;
                out.data[2 * HW + i] += d;
            }
        }
    }

    // =================== POST ===================

    void apply_post_filter(const Tensor& in, Tensor& out,
        FilterState& st, const FilterOptions& opt)
    {
        int W = in.W, H = in.H;
        int HW = H * W;

        bool do_dehalo = opt.dehalo && opt.dehalo_strength > 0.f;
        bool do_sharp = opt.post_sharp && opt.post_sharp_amount > 0.f;

        if (!do_dehalo && !do_sharp) {
            out = in;
            return;
        }

        if (out.C != in.C || out.H != in.H || out.W != in.W)
            out = Tensor(in.C, in.H, in.W);
        std::memcpy(out.data.data(), in.data.data(), in.data.size() * sizeof(float));

        // --- 1. DeHalo (первым!) — утончает границы и убирает ореолы ---
        if (do_dehalo) {
            dehalo_inplace(out, opt.dehalo_strength, opt.dehalo_dark_bias);
        }

        // --- 2. Post-sharp (после dehalo, по уже тонким границам) ---
        if (!do_sharp) return;

        std::vector<float> lum(HW), blur(HW), grad(HW);
        for (int i = 0; i < HW; ++i)
            lum[i] = luma3(out.data[i], out.data[HW + i], out.data[2 * HW + i]);
        blur5x5(lum.data(), blur.data(), H, W);
        gradient_mag(lum.data(), grad.data(), H, W);

        const float amt = opt.post_sharp_amount;
        const float guard = opt.post_sharp_halo_guard;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < HW; ++i) {
            float detail = lum[i] - blur[i];
            float halo_gate = 1.f - smoothstep(grad[i], guard, guard * 3.f);
            float tex_gate = smoothstep(std::fabs(detail), 0.f, 0.02f);
            float d = amt * halo_gate * tex_gate * detail;
            out.data[i] += d;
            out.data[HW + i] += d;
            out.data[2 * HW + i] += d;
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)out.data.size(); ++i) {
            if (out.data[i] < 0.f) out.data[i] = 0.f;
            else if (out.data[i] > 1.f) out.data[i] = 1.f;
        }
    }

}