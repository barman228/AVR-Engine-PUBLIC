#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "mvmod10_engine.h"
#include <stdexcept>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define MV_AVX2 1
#else
#define MV_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mvmod10 {

    // ======================= примитивы =======================

    static void conv2d_3x3(const Tensor& x, const Weight& w, const Weight& b, Tensor& y) {
        int Ci = x.C, H = x.H, W = x.W, Co = w.shape[0];
        // Переиспользуем буфер, если размеры совпадают (избегаем alloc + memset).
        // Все позиции выхода записываются полностью — init не нужен.
        if (y.C != Co || y.H != H || y.W != W) {
            y = Tensor(Co, H, W);
        }

#pragma omp parallel for schedule(static)
        for (int co = 0; co < Co; ++co) {
            const float* wco = &w.data[(size_t)co * Ci * 9];
            float bias = b.data[co];
            float* yco = y.ch(co);

            for (int yy = 0; yy < H; ++yy) {
                int is_tb = (yy == 0) || (yy == H - 1);
                for (int xx = 0; xx < W; ++xx) {
                    bool border = is_tb || (xx == 0) || (xx == W - 1);
                    if (!border) continue;
                    float acc = bias;
                    for (int ci = 0; ci < Ci; ++ci) {
                        const float* xci = x.ch(ci);
                        const float* wc = wco + (size_t)ci * 9;
                        for (int ky = -1; ky <= 1; ++ky) {
                            int ys = yy + ky;
                            if (ys < 0 || ys >= H) continue;
                            for (int kx = -1; kx <= 1; ++kx) {
                                int xs = xx + kx;
                                if (xs < 0 || xs >= W) continue;
                                acc += xci[ys * W + xs] * wc[(ky + 1) * 3 + (kx + 1)];
                            }
                        }
                    }
                    yco[yy * W + xx] = acc;
                }
            }

            for (int yy = 1; yy < H - 1; ++yy) {
                int xx = 1;
#if MV_AVX2
                for (; xx + 8 <= W - 1; xx += 8) {
                    __m256 acc = _mm256_set1_ps(bias);
                    for (int ci = 0; ci < Ci; ++ci) {
                        const float* xci = x.ch(ci);
                        const float* wc = wco + (size_t)ci * 9;
                        for (int ky = -1; ky <= 1; ++ky) {
                            int yoff = (yy + ky) * W + xx;
                            for (int kx = -1; kx <= 1; ++kx) {
                                __m256 xv = _mm256_loadu_ps(xci + yoff + kx);
                                __m256 wv = _mm256_set1_ps(wc[(ky + 1) * 3 + (kx + 1)]);
                                acc = _mm256_fmadd_ps(xv, wv, acc);
                            }
                        }
                    }
                    _mm256_storeu_ps(yco + yy * W + xx, acc);
                }
#endif
                for (; xx < W - 1; ++xx) {
                    float acc = bias;
                    for (int ci = 0; ci < Ci; ++ci) {
                        const float* xci = x.ch(ci);
                        const float* wc = wco + (size_t)ci * 9;
                        for (int ky = -1; ky <= 1; ++ky)
                            for (int kx = -1; kx <= 1; ++kx)
                                acc += xci[(yy + ky) * W + (xx + kx)]
                                * wc[(ky + 1) * 3 + (kx + 1)];
                    }
                    yco[yy * W + xx] = acc;
                }
            }
        }
    }


    static void depthwise_conv3x3(const Tensor& x, const Weight& w, const Weight& b,
        Tensor& y, int dilation)
    {
        int C = x.C, H = x.H, W = x.W;
        if (y.C != C || y.H != H || y.W != W) {
            y = Tensor(C, H, W);
        }
        int d = dilation;
        int border = d;

#pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            const float* wc = &w.data[(size_t)c * 9];
            float bias = b.data[c];
            const float* xc = x.ch(c);
            float* yc = y.ch(c);

            for (int yy = 0; yy < H; ++yy) {
                int is_top_bottom = (yy < border) || (yy >= H - border);
                for (int xx = 0; xx < W; ++xx) {
                    bool is_border = is_top_bottom || (xx < border) || (xx >= W - border);
                    if (!is_border) continue;
                    float acc = bias;
                    for (int ky = -1; ky <= 1; ++ky) {
                        int ys = yy + ky * d;
                        if (ys < 0 || ys >= H) continue;
                        for (int kx = -1; kx <= 1; ++kx) {
                            int xs = xx + kx * d;
                            if (xs < 0 || xs >= W) continue;
                            acc += xc[ys * W + xs] * wc[(ky + 1) * 3 + (kx + 1)];
                        }
                    }
                    yc[yy * W + xx] = acc;
                }
            }

            for (int yy = border; yy < H - border; ++yy) {
#if MV_AVX2
                int xx = border;
                for (; xx + 8 <= W - border; xx += 8) {
                    __m256 acc = _mm256_set1_ps(bias);
                    for (int ky = -1; ky <= 1; ++ky) {
                        int yoff = (yy + ky * d) * W + xx;
                        for (int kx = -1; kx <= 1; ++kx) {
                            __m256 xv = _mm256_loadu_ps(xc + yoff + kx * d);
                            __m256 wv = _mm256_set1_ps(wc[(ky + 1) * 3 + (kx + 1)]);
                            acc = _mm256_fmadd_ps(xv, wv, acc);
                        }
                    }
                    _mm256_storeu_ps(yc + yy * W + xx, acc);
                }
                for (; xx < W - border; ++xx) {
                    float acc = bias;
                    for (int ky = -1; ky <= 1; ++ky) {
                        int ys = yy + ky * d;
                        for (int kx = -1; kx <= 1; ++kx) {
                            int xs = xx + kx * d;
                            acc += xc[ys * W + xs] * wc[(ky + 1) * 3 + (kx + 1)];
                        }
                    }
                    yc[yy * W + xx] = acc;
                }
#else
                for (int xx = border; xx < W - border; ++xx) {
                    float acc = bias;
                    for (int ky = -1; ky <= 1; ++ky) {
                        int ys = yy + ky * d;
                        for (int kx = -1; kx <= 1; ++kx) {
                            int xs = xx + kx * d;
                            acc += xc[ys * W + xs] * wc[(ky + 1) * 3 + (kx + 1)];
                        }
                    }
                    yc[yy * W + xx] = acc;
                }
#endif
            }
        }
    }


    static void conv2d_1x1(const Tensor& x, const Weight& w, const Weight& b, Tensor& y) {
        int Ci = x.C, H = x.H, W = x.W, Co = w.shape[0];
        if (y.C != Co || y.H != H || y.W != W) {
            y = Tensor(Co, H, W);
        }
        int HW = H * W;

#pragma omp parallel for schedule(static)
        for (int co = 0; co < Co; ++co) {
            const float* wco = &w.data[(size_t)co * Ci];
            float bias = b.data[co];
            float* yco = y.ch(co);

            // 1×1 всё равно требует init — все outputs накапливаются через +=
            for (int i = 0; i < HW; ++i) yco[i] = bias;

#if MV_AVX2
            for (int ci = 0; ci < Ci; ++ci) {
                const float* xci = x.ch(ci);
                __m256 wv = _mm256_set1_ps(wco[ci]);
                int i = 0;
                for (; i + 8 <= HW; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xci + i);
                    __m256 yv = _mm256_loadu_ps(yco + i);
                    yv = _mm256_fmadd_ps(xv, wv, yv);
                    _mm256_storeu_ps(yco + i, yv);
                }
                for (; i < HW; ++i) yco[i] += xci[i] * wco[ci];
            }
#else
            for (int ci = 0; ci < Ci; ++ci) {
                const float* xci = x.ch(ci);
                float wc = wco[ci];
                for (int i = 0; i < HW; ++i) yco[i] += xci[i] * wc;
            }
#endif
        }
    }

    static inline void relu_inplace(Tensor& t) {
        for (auto& v : t.data) if (v < 0.f) v = 0.f;
    }

    static void prelu_inplace(Tensor& t, const Weight& slope) {
        int C = t.C, HW = t.H * t.W;
        for (int c = 0; c < C; ++c) {
            float s = slope.data[c];
            float* p = t.ch(c);
            for (int i = 0; i < HW; ++i) if (p[i] < 0.f) p[i] *= s;
        }
    }

    static Tensor bilinear_upsample(const Tensor& x, int H_out, int W_out) {
        Tensor y(x.C, H_out, W_out);
        float ry = (float)x.H / H_out;
        float rx = (float)x.W / W_out;
        for (int c = 0; c < x.C; ++c) {
            const float* xc = x.ch(c);
            float* yc = y.ch(c);
            for (int dy = 0; dy < H_out; ++dy) {
                float sy = (dy + 0.5f) * ry - 0.5f;
                int y0 = (int)std::floor(sy);
                float wy = sy - y0;
                int y1 = y0 + 1;
                if (y0 < 0) { y0 = 0; y1 = 0; wy = 0.f; }
                if (y1 >= x.H) { y1 = x.H - 1; y0 = x.H - 1; wy = 0.f; }
                for (int dx = 0; dx < W_out; ++dx) {
                    float sx = (dx + 0.5f) * rx - 0.5f;
                    int x0 = (int)std::floor(sx);
                    float wx = sx - x0;
                    int x1 = x0 + 1;
                    if (x0 < 0) { x0 = 0; x1 = 0; wx = 0.f; }
                    if (x1 >= x.W) { x1 = x.W - 1; x0 = x.W - 1; wx = 0.f; }
                    float v00 = xc[y0 * x.W + x0];
                    float v01 = xc[y0 * x.W + x1];
                    float v10 = xc[y1 * x.W + x0];
                    float v11 = xc[y1 * x.W + x1];
                    yc[dy * W_out + dx] =
                        (1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v01 +
                        wy * (1 - wx) * v10 + wy * wx * v11;
                }
            }
        }
        return y;
    }

    // ======================= tile features =======================

    static void compute_tile_features(const Tensor& lr, const Tensor* prev_lr,
        int T, std::vector<float>& out_feat)
    {
        int H = lr.H, W = lr.W;
        int H_t = H / T, W_t = W / T, N = H_t * W_t;
        int HW = H * W;
        out_feat.assign((size_t)N * 4, 0.f);

        std::vector<float> gray(HW);
        for (int i = 0; i < HW; ++i)
            gray[i] = 0.299f * lr.data[i] + 0.587f * lr.data[HW + i] + 0.114f * lr.data[2 * HW + i];

        // motion = mean(|lr - prev_lr|) по каналам
        std::vector<float> motion(HW, 0.f);
        if (prev_lr) {
            for (int i = 0; i < HW; ++i) {
                float d = 0.f;
                for (int c = 0; c < 3; ++c)
                    d += std::fabs(lr.data[c * HW + i] - prev_lr->data[c * HW + i]);
                motion[i] = d * (1.f / 3.f);
            }
        }

        // Sobel magnitude
        std::vector<float> grad(HW, 0.f);
        static const int SX[9] = { -1,0,1,-2,0,2,-1,0,1 };
        static const int SY[9] = { -1,-2,-1,0,0,0,1,2,1 };
        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                float gx = 0.f, gy = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    int ys = yy + ky; if (ys < 0 || ys >= H) continue;
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xs = xx + kx; if (xs < 0 || xs >= W) continue;
                        float v = gray[ys * W + xs];
                        int idx = (ky + 1) * 3 + (kx + 1);
                        gx += v * SX[idx];
                        gy += v * SY[idx];
                    }
                }
                grad[yy * W + xx] = std::sqrt(gx * gx + gy * gy);
            }
        }

        // Avg pool по тайлам T×T
        float inv = 1.f / (float)(T * T);
        for (int ty = 0; ty < H_t; ++ty) {
            for (int tx = 0; tx < W_t; ++tx) {
                float sm = 0.f, sm2 = 0.f, sg = 0.f, sgr = 0.f;
                for (int dy = 0; dy < T; ++dy) {
                    int y = ty * T + dy;
                    for (int dx = 0; dx < T; ++dx) {
                        int x = tx * T + dx;
                        float m = motion[y * W + x];
                        sm += m; sm2 += m * m;
                        sg += gray[y * W + x];
                        sgr += grad[y * W + x];
                    }
                }
                float m_mean = sm * inv;
                float m_var = std::max(0.f, sm2 * inv - m_mean * m_mean);
                float m_std = std::sqrt(m_var);
                int n = ty * W_t + tx;
                out_feat[n * 4 + 0] = m_mean;
                out_feat[n * 4 + 1] = m_std;
                out_feat[n * 4 + 2] = sg * inv;
                out_feat[n * 4 + 3] = sgr * inv;
            }
        }
    }

    // ======================= router =======================

    static void router_forward(const std::vector<float>& tile_feat, int N,
        const Weight& bn_w, const Weight& bn_b,
        const Weight& bn_mean, const Weight& bn_var,
        const Weight& fc1_w, const Weight& fc1_b,
        const Weight& fc2_w, const Weight& fc2_b,
        std::vector<float>& out_w)
    {
        const float eps = 1e-5f;
        out_w.assign((size_t)N * 3, 0.f);
        for (int n = 0; n < N; ++n) {
            float b[4];
            for (int i = 0; i < 4; ++i) {
                float v = tile_feat[n * 4 + i];
                b[i] = (v - bn_mean.data[i]) / std::sqrt(bn_var.data[i] + eps)
                    * bn_w.data[i] + bn_b.data[i];
            }
            float h[16];
            for (int j = 0; j < 16; ++j) {
                float s = fc1_b.data[j];
                for (int i = 0; i < 4; ++i) s += b[i] * fc1_w.data[j * 4 + i];
                h[j] = s > 0 ? s : 0;
            }
            float logits[3];
            for (int k = 0; k < 3; ++k) {
                float s = fc2_b.data[k];
                for (int j = 0; j < 16; ++j) s += h[j] * fc2_w.data[k * 16 + j];
                logits[k] = s;
            }
            float mx = std::max({ logits[0], logits[1], logits[2] });
            float sum = 0.f;
            for (int k = 0; k < 3; ++k) { logits[k] = std::exp(logits[k] - mx); sum += logits[k]; }
            for (int k = 0; k < 3; ++k) out_w[n * 3 + k] = logits[k] / sum;
        }
    }

    // ======================= temporal =======================

    static Tensor temporal_blend(const Tensor& f, const Tensor* prev,
        const Weight& pw, const Weight& pb)
    {
        if (!prev) return f;
        int C = f.C, H = f.H, W = f.W;
        Tensor out(C, H, W);

#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float acc = pb.data[0];
                for (int c = 0; c < C; ++c)
                    acc += f.at(c, y, x)[0] * pw.data[c];
                for (int c = 0; c < C; ++c)
                    acc += prev->at(c, y, x)[0] * pw.data[C + c];
                float w = 1.f / (1.f + std::exp(-acc));
                float iw = 1.f - w;
                for (int c = 0; c < C; ++c)
                    out.at(c, y, x)[0] = w * f.at(c, y, x)[0] + iw * prev->at(c, y, x)[0];
            }
        }
        return out;
    }

    // ======================= DS-block =======================

    // Записывает результат DS-блока в out (должен быть правильных размеров).
    // Промежуточные dw/pw переиспользуются через thread_local.
    // SE-скейлинг и residual-scale α фьюзятся в один проход.
    static void ds_block_into(const Tensor& x, Tensor& out, int dilation,
        const Weight& w_dw, const Weight& b_dw,
        const Weight& w_pw, const Weight& b_pw,
        const Weight& prelu_slope,
        const Weight* w_se1, const Weight* b_se1,
        const Weight* w_se2, const Weight* b_se2,
        const Weight& alpha)
    {
        static thread_local Tensor tl_dw, tl_pw;

        if (tl_dw.C != x.C || tl_dw.H != x.H || tl_dw.W != x.W)
            tl_dw = Tensor(x.C, x.H, x.W);
        if (tl_pw.C != x.C || tl_pw.H != x.H || tl_pw.W != x.W)
            tl_pw = Tensor(x.C, x.H, x.W);

        depthwise_conv3x3(x, w_dw, b_dw, tl_dw, dilation);
        prelu_inplace(tl_dw, prelu_slope);
        conv2d_1x1(tl_dw, w_pw, b_pw, tl_pw);

        // --- SE: считаем per-channel scales в маленький локальный массив ---
        bool has_se = (w_se1 && w_se2);
        std::vector<float> se_scale;
        if (has_se) {
            int C = tl_pw.C, HW = tl_pw.H * tl_pw.W;
            int hid = w_se1->shape[0];
            se_scale.resize(C);

            std::vector<float> g(C, 0.f);
            for (int c = 0; c < C; ++c) {
                const float* p = tl_pw.ch(c);
                float s = 0.f;
                for (int i = 0; i < HW; ++i) s += p[i];
                g[c] = s / HW;
            }
            std::vector<float> h(hid);
            for (int j = 0; j < hid; ++j) {
                float s = b_se1->data[j];
                for (int c = 0; c < C; ++c) s += g[c] * w_se1->data[j * C + c];
                h[j] = s > 0 ? s : 0;
            }
            for (int c = 0; c < C; ++c) {
                float s = b_se2->data[c];
                for (int j = 0; j < hid; ++j) s += h[j] * w_se2->data[c * hid + j];
                se_scale[c] = 1.f / (1.f + std::exp(-s));
            }
        }

        // --- Residual + SE + α в одном проходе ---
        if (out.C != x.C || out.H != x.H || out.W != x.W) {
            out = Tensor(x.C, x.H, x.W);
        }
        float a = alpha.data[0];
        for (int c = 0; c < x.C; ++c) {
            float scale = has_se ? a * se_scale[c] : a;
            const float* xp = x.ch(c);
            const float* yp = tl_pw.ch(c);
            float* op = out.ch(c);
            int HW = x.H * x.W;
            for (int i = 0; i < HW; ++i) op[i] = xp[i] + scale * yp[i];
        }
    }

    static Tensor ds_block(const Tensor& x, int dilation,
        const Weight& w_dw, const Weight& b_dw,
        const Weight& w_pw, const Weight& b_pw,
        const Weight& prelu_slope,
        const Weight* w_se1, const Weight* b_se1,
        const Weight* w_se2, const Weight* b_se2,
        const Weight& alpha)
    {
        Tensor dw, pw;
        depthwise_conv3x3(x, w_dw, b_dw, dw, dilation);
        prelu_inplace(dw, prelu_slope);
        conv2d_1x1(dw, w_pw, b_pw, pw);

        if (w_se1 && w_se2) {
            int C = pw.C, HW = pw.H * pw.W;
            int hid = w_se1->shape[0];
            std::vector<float> g(C, 0.f);
            for (int c = 0; c < C; ++c) {
                const float* p = pw.ch(c);
                float s = 0.f;
                for (int i = 0; i < HW; ++i) s += p[i];
                g[c] = s / HW;
            }
            std::vector<float> h(hid);
            for (int j = 0; j < hid; ++j) {
                float s = b_se1->data[j];
                for (int c = 0; c < C; ++c) s += g[c] * w_se1->data[j * C + c];
                h[j] = s > 0 ? s : 0;
            }
            std::vector<float> sc(C);
            for (int c = 0; c < C; ++c) {
                float s = b_se2->data[c];
                for (int j = 0; j < hid; ++j) s += h[j] * w_se2->data[c * hid + j];
                sc[c] = 1.f / (1.f + std::exp(-s));
            }
            for (int c = 0; c < C; ++c) {
                float* p = pw.ch(c);
                float s = sc[c];
                for (int i = 0; i < HW; ++i) p[i] *= s;
            }
        }

        Tensor out(x.C, x.H, x.W);
        float a = alpha.data[0];
        for (int c = 0; c < x.C; ++c) {
            const float* xp = x.ch(c);
            const float* yp = pw.ch(c);
            float* op = out.ch(c);
            int HW = x.H * x.W;
            for (int i = 0; i < HW; ++i) op[i] = xp[i] + a * yp[i];
        }
        return out;
    }

    // ======================= expert =======================

    // Прогоняет блоки эксперта от x до out. Внутри — ping-pong двух
    // thread_local буферов: ни одной аллокации на блок.
    void MVNNModel::run_expert_into(const Tensor& x, Tensor& out,
        const char* prefix, int n_blocks) const
    {
        static thread_local Tensor tl_a, tl_b;

        if (tl_a.C != x.C || tl_a.H != x.H || tl_a.W != x.W)
            tl_a = Tensor(x.C, x.H, x.W);
        if (tl_b.C != x.C || tl_b.H != x.H || tl_b.W != x.W)
            tl_b = Tensor(x.C, x.H, x.W);

        // x -> tl_a (копируем вход в буфер, потому что дальше ping-pong)
        std::memcpy(tl_a.data.data(), x.data.data(), x.data.size() * sizeof(float));

        Tensor* cur = &tl_a;
        Tensor* next = &tl_b;

        for (int i = 0; i < n_blocks; ++i) {
            std::string base = std::string(prefix) + ".blocks."
                + std::to_string(i) + ".";
            int dilation = (i % 2 == 0) ? 1 : 2;

            auto get = [&](const std::string& k) -> const Weight& {
                auto it = W.find(k);
                if (it == W.end()) throw std::runtime_error("missing weight: " + k);
                return it->second;
                };

            const Weight* w_se1 = nullptr; const Weight* b_se1 = nullptr;
            const Weight* w_se2 = nullptr; const Weight* b_se2 = nullptr;
            auto it1 = W.find(base + "se_fc1.weight");
            if (it1 != W.end()) {
                w_se1 = &it1->second;
                b_se1 = &W.at(base + "se_fc1.bias");
                w_se2 = &W.at(base + "se_fc2.weight");
                b_se2 = &W.at(base + "se_fc2.bias");
            }

            ds_block_into(*cur, *next, dilation,
                get(base + "dw.weight"), get(base + "dw.bias"),
                get(base + "pw.weight"), get(base + "pw.bias"),
                get(base + "act.weight"),
                w_se1, b_se1, w_se2, b_se2,
                get(base + "alpha"));

            std::swap(cur, next);
        }

        // final = *cur. Копируем в out (у него правильный размер — проверит вызывающий).
        if (out.C != cur->C || out.H != cur->H || out.W != cur->W) {
            out = Tensor(cur->C, cur->H, cur->W);
        }
        std::memcpy(out.data.data(), cur->data.data(),
            cur->data.size() * sizeof(float));
    }

    // ======================= loader =======================

    static bool read_u32(FILE* f, uint32_t& v) { return std::fread(&v, 4, 1, f) == 1; }
    static bool read_bytes(FILE* f, void* p, size_t n) { return std::fread(p, 1, n, f) == n; }

    bool MVNNModel::load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;

        char magic[4];
        if (!read_bytes(f, magic, 4)) { std::fclose(f); return false; }
        if (std::memcmp(magic, "MVNN", 4) != 0) { std::fclose(f); return false; }

        uint32_t version, header_size, n_params, weights_size;
        if (!read_u32(f, version) || version != 1) { std::fclose(f); return false; }
        if (!read_u32(f, header_size)) { std::fclose(f); return false; }
        if (!read_u32(f, n_params)) { std::fclose(f); return false; }
        if (!read_u32(f, weights_size)) { std::fclose(f); return false; }

        auto rdu = [&](uint32_t& v) { if (!read_u32(f, v)) throw std::runtime_error("short header"); };

        rdu(in_ch); rdu(feat); rdu(tile);
        rdu(e0_blocks); rdu(e1_blocks); rdu(e2_blocks); rdu(e2_use_se);
        rdu(max_scale); rdu(default_scale); rdu(n_scales);

        scale_ids.resize(n_scales);
        for (auto& s : scale_ids) rdu(s);

        scale_ch_offsets.resize(n_scales);
        for (auto& p : scale_ch_offsets) {
            uint32_t a, b; rdu(a); rdu(b);
            p = { a, b };
        }

        uint32_t n_layers; rdu(n_layers);

        std::vector<std::string> names(n_layers);
        for (uint32_t i = 0; i < n_layers; ++i) {
            uint32_t nl; rdu(nl);
            names[i].resize(nl);
            if (!read_bytes(f, names[i].data(), nl)) throw std::runtime_error("short name");
            uint32_t ndim; rdu(ndim);
            std::vector<int> shape(ndim);
            for (auto& d : shape) { uint32_t dd; rdu(dd); d = (int)dd; }
            uint32_t dt; rdu(dt);
            if (dt != 0) throw std::runtime_error("only fp32 supported");
            W[names[i]].shape = std::move(shape);
        }

        // Пропускаем padding до конца заголовка
        long cur = std::ftell(f);
        long skip = (long)header_size - cur;
        if (skip > 0) std::fseek(f, skip, SEEK_CUR);

        // Читаем веса
        for (auto& name : names) {
            auto& wt = W[name];
            wt.data.resize(wt.numel());
            if (!read_bytes(f, wt.data.data(), wt.data.size() * 4))
                throw std::runtime_error("short weights for " + name);
        }

        std::fclose(f);
        return true;
    }

    // ======================= hard-routing helpers =======================

    static void pack_tiles(const Tensor& src,
        const std::vector<std::pair<int, int>>& tiles,
        int T, int halo, Tensor& packed)
    {
        int C = src.C;
        int H_p = src.H, W_p = src.W;
        int P = T + 2 * halo;
        int N = (int)tiles.size();
        packed = Tensor(C, N * P, P);

        for (int n = 0; n < N; ++n) {
            int ty = tiles[n].first, tx = tiles[n].second;
            int y0 = ty * T - halo;
            int x0 = tx * T - halo;
            for (int c = 0; c < C; ++c) {
                const float* sp = src.ch(c);
                float* dp = packed.ch(c) + (size_t)n * P * P;
                for (int dy = 0; dy < P; ++dy) {
                    int y = y0 + dy;
                    if (y < 0) y = 0;
                    if (y >= H_p) y = H_p - 1;
                    const float* srow = sp + (size_t)y * W_p;
                    float* drow = dp + (size_t)dy * P;
                    for (int dx = 0; dx < P; ++dx) {
                        int x = x0 + dx;
                        if (x < 0) x = 0;
                        if (x >= W_p) x = W_p - 1;
                        drow[dx] = srow[x];
                    }
                }
            }
        }
    }

    static void unpack_tiles(const Tensor& packed, Tensor& dst,
        const std::vector<std::pair<int, int>>& tiles,
        int T, int halo)
    {
        int C = dst.C;
        int P = T + 2 * halo;
        int N = (int)tiles.size();
        for (int n = 0; n < N; ++n) {
            int ty = tiles[n].first, tx = tiles[n].second;
            for (int c = 0; c < C; ++c) {
                const float* sp = packed.ch(c) + (size_t)n * P * P;
                for (int dy = 0; dy < T; ++dy) {
                    const float* srow = sp + (size_t)(dy + halo) * P + halo;
                    float* drow = dst.at(c, ty * T + dy, tx * T);
                    for (int dx = 0; dx < T; ++dx) drow[dx] = srow[dx];
                }
            }
        }
    }

    // ======================= forward =======================

    Tensor MVNNModel::forward(const Tensor& lr_in, const Tensor* prev_lr_in,
        const Tensor* state_in, Tensor* state_out,
        int scale, float strength, bool clamp_out) const
    {
        int H = lr_in.H, W_in = lr_in.W;       // ← W_in, чтобы не затенять this->W
        int T = (int)tile;
        int H_p = ((H + T - 1) / T) * T;
        int W_p = ((W_in + T - 1) / T) * T;

        Tensor lr, prev_p;
        if (H_p != H || W_p != W_in) {
            lr = Tensor(3, H_p, W_p);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < H_p; ++y) {
                    int ys = std::min(y, H - 1);
                    for (int x = 0; x < W_p; ++x) {
                        int xs = std::min(x, W_in - 1);
                        lr.at(c, y, x)[0] = lr_in.at(c, ys, xs)[0];
                    }
                }
            if (prev_lr_in) {
                prev_p = Tensor(3, H_p, W_p);
                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < H_p; ++y) {
                        int ys = std::min(y, H - 1);
                        for (int x = 0; x < W_p; ++x) {
                            int xs = std::min(x, W_in - 1);
                            prev_p.at(c, y, x)[0] = prev_lr_in->at(c, ys, xs)[0];
                        }
                    }
            }
        }
        else {
            lr = lr_in;
            if (prev_lr_in) prev_p = *prev_lr_in;
        }

        // 1. Head
        Tensor f;
        conv2d_3x3(lr, W.at("head.weight"), W.at("head.bias"), f);
        relu_inplace(f);

        // 2. Temporal
        Tensor f_t = temporal_blend(f, state_in, W.at("temporal.proj.weight"),
            W.at("temporal.proj.bias"));
        if (state_out) *state_out = f_t;

        // 3. Router
        int H_t = H_p / T, W_t = W_p / T, N = H_t * W_t;
        std::vector<float> tile_feat;
        compute_tile_features(lr, prev_lr_in ? &prev_p : nullptr, T, tile_feat);

        std::vector<float> tile_w;
        router_forward(tile_feat, N,
            W.at("router.norm.weight"), W.at("router.norm.bias"),
            W.at("router.norm.running_mean"), W.at("router.norm.running_var"),
            W.at("router.fc1.weight"), W.at("router.fc1.bias"),
            W.at("router.fc2.weight"), W.at("router.fc2.bias"),
            tile_w);

        // 4. Experts — HARD ROUTING (per-tile)
        // Каждый тайл идёт только в свой эксперт (argmax от softmax).
        // Обрабатываем тайлы ПООТДЕЛЬНОСТИ, чтобы свёртки не "пробивали"
        // границы между упакованными тайлами в batched-режиме.
        // OpenMP параллелит по тайлам — суммарный выигрыш сохраняется.

        const int HALO = 3;   // перекрытие контекста для бесшовной стыковки
        const int P = T + 2 * HALO;

        // Классифицируем тайлы по экспертам
        struct TileJob { int ty, tx, expert; };
        std::vector<TileJob> jobs;
        jobs.reserve(N);
        for (int n = 0; n < N; ++n) {
            float w0 = tile_w[n * 3 + 0], w1 = tile_w[n * 3 + 1], w2 = tile_w[n * 3 + 2];
            int e = (w2 > w0 && w2 > w1) ? 2 : (w1 > w0 ? 1 : 0);
            jobs.push_back({ n / W_t, n % W_t, e });
        }

        Tensor f_out(feat, H_p, W_p);

        const char* expert_names[3] = { "e0", "e1", "e2" };
        int         expert_blocks[3] = { (int)e0_blocks, (int)e1_blocks, (int)e2_blocks };

#pragma omp parallel for schedule(dynamic, 4)
        for (int idx = 0; idx < (int)jobs.size(); ++idx) {
            const auto& j = jobs[idx];
            int y0 = j.ty * T - HALO;
            int x0 = j.tx * T - HALO;

            // thread_local — каждый OMP-поток имеет свои буферы,
            // аллокаций внутри цикла не происходит.
            static thread_local Tensor tl_patch, tl_exp;

            if (tl_patch.C != (int)feat || tl_patch.H != P || tl_patch.W != P)
                tl_patch = Tensor((int)feat, P, P);
            if (tl_exp.C != (int)feat || tl_exp.H != P || tl_exp.W != P)
                tl_exp = Tensor((int)feat, P, P);

            // --- Вырезаем патч [feat, P, P] с halo ---
            for (int c = 0; c < (int)feat; ++c) {
                const float* sp = f_t.ch(c);
                float* dp = tl_patch.ch(c);
                for (int dy = 0; dy < P; ++dy) {
                    int y = y0 + dy;
                    if (y < 0) y = 0;
                    if (y >= H_p) y = H_p - 1;
                    const float* srow = sp + (size_t)y * W_p;
                    float* drow = dp + (size_t)dy * P;
                    for (int dx = 0; dx < P; ++dx) {
                        int x = x0 + dx;
                        if (x < 0) x = 0;
                        if (x >= W_p) x = W_p - 1;
                        drow[dx] = srow[x];
                    }
                }
            }

            // --- Прогоняем через соответствующий эксперт ---
            run_expert_into(tl_patch, tl_exp,
                expert_names[j.expert],
                expert_blocks[j.expert]);

            // --- Записываем центральные T×T в f_out ---
            for (int c = 0; c < (int)feat; ++c) {
                const float* sp = tl_exp.ch(c);
                float* dp = f_out.ch(c);
                for (int dy = 0; dy < T; ++dy) {
                    const float* srow = sp + (size_t)(dy + HALO) * P + HALO;
                    float* drow = dp + (size_t)(j.ty * T + dy) * W_p + j.tx * T;
                    for (int dx = 0; dx < T; ++dx) drow[dx] = srow[dx];
                }
            }
        }

        // 6. Scale embedding
        const Weight& se = W.at("scale_embed.weight");
        for (int c = 0; c < (int)feat; ++c) {
            float e = se.data[(size_t)scale * feat + c];
            float* p = f_out.ch(c);
            for (int i = 0; i < H_p * W_p; ++i) p[i] += e;
        }

        // 7. Tail
        Tensor y_all;
        conv2d_3x3(f_out, W.at("tail.weight"), W.at("tail.bias"), y_all);

        // 8. Slice + PixelShuffle
        int s_start = 0, s_end = 0;
        for (size_t i = 0; i < scale_ids.size(); ++i) {
            if ((int)scale_ids[i] == scale) {
                s_start = (int)scale_ch_offsets[i].first;
                s_end = (int)scale_ch_offsets[i].second;
                break;
            }
        }
        int r = scale;
        int Co = (s_end - s_start) / (r * r);
        Tensor y(Co, H_p * r, W_p * r);
        for (int c = 0; c < Co; ++c) {
            for (int dy = 0; dy < r; ++dy) {
                for (int dx = 0; dx < r; ++dx) {
                    int src = s_start + c * r * r + dy * r + dx;
                    for (int yy = 0; yy < H_p; ++yy)
                        for (int xx = 0; xx < W_p; ++xx)
                            y.at(c, yy * r + dy, xx * r + dx)[0] = y_all.at(src, yy, xx)[0];
                }
            }
        }

        // 9. Crop padding
        int Ho = H * r, Wo = W_in * r;
        Tensor yc(Co, Ho, Wo);
        for (int c = 0; c < Co; ++c)
            for (int yy = 0; yy < Ho; ++yy)
                std::memcpy(yc.at(c, yy, 0), y.at(c, yy, 0), Wo * sizeof(float));

        // 10. Strength
        if (strength != 1.0f) {
            Tensor base = bilinear_upsample(lr_in, Ho, Wo);
            for (size_t i = 0; i < yc.data.size(); ++i)
                yc.data[i] = base.data[i] + strength * (yc.data[i] - base.data[i]);
        }

        // 11. Clamp
        if (clamp_out) {
            for (auto& v : yc.data) {
                if (v < 0.f) v = 0.f;
                else if (v > 1.f) v = 1.f;
            }
        }

        return yc;
    }

}