#pragma once
#include "mvmod10_engine.h"

namespace mvmod10 {

    struct FilterOptions {
        // --- Temporal denoise ---
        bool  temporal_denoise = false;
        float denoise_strength = 0.6f;
        float denoise_motion_thresh = 0.02f;

        // --- Pre-sharp ---
        bool  pre_sharp = false;
        float pre_sharp_amount = 0.20f;

        // --- Post-filters (порядок применения: dehalo → post_sharp) ---
        // DeHalo: убирает «жирные» границы и чёрные/светлые ореолы,
        // которые SR-модель оставляет вокруг контуров. Работает в luma.
        bool  dehalo = false;
        float dehalo_strength = 0.5f;      // 0..1, основной регулятор
        float dehalo_dark_bias = 0.3f;     // 0..1, дополнительная сила именно для тёмных ореолов

        // Post-sharp: структура с подавлением halo
        bool  post_sharp = false;
        float post_sharp_amount = 0.30f;
        float post_sharp_halo_guard = 0.06f;
    };

    struct FilterState {
        Tensor prev_luma;      // [1,H,W] — luma предыдущего СЫРОГО кадра (для motion)
        Tensor history_luma;   // [1,H,W] — накопленный denoise luma
        Tensor history_rgb;    // [3,H,W] — накопленный denoise RGB
        int last_W = 0, last_H = 0;
        bool initialized = false;

        void reset() { initialized = false; last_W = last_H = 0; }

        void ensure(int W, int H) {
            if (last_W != W || last_H != H) {
                prev_luma = Tensor(1, H, W);
                history_luma = Tensor(1, H, W);
                history_rgb = Tensor(3, H, W);
                last_W = W; last_H = H;
                initialized = false;
            }
        }
    };

    // Применить pre-фильтр: denoise (если включён) + pre-sharp.
    // in → out (out будет переназначен нужного размера).
    void apply_pre_filter(const Tensor& in, Tensor& out,
        FilterState& st, const FilterOptions& opt);

    // Применить post-фильтр: post-sharp с halo-защитой.
    void apply_post_filter(const Tensor& in, Tensor& out,
        FilterState& st, const FilterOptions& opt);

}