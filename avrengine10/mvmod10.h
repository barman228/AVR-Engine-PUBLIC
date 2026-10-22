#pragma once
#include <string>

namespace mvmod10 {

    struct VideoOptions {
        int   threads = 0;                  // 0 = auto (hardware_concurrency)
        float strength = 1.0f;              // 0.0 = bilinear, 1.0 = как обучено, 1.4 = oversharpen
        bool  stateful = true;              // переносить temporal state между кадрами
        float scene_cut_thresh = 0.25f;     // средний |diff| для сброса state
        bool  verbose = true;
        // Если > 0 — переопределяет fps, полученный из ffprobe.
        double fps_override = 0.0;

        // Post/Pre filters
        bool  temporal_denoise = false;
        float denoise_strength = 0.6f;
        float denoise_motion_thresh = 0.02f;
        bool  pre_sharp = false;
        float pre_sharp_amount = 0.20f;

        // DeHalo: убирает «жирные» границы и чёрные/светлые ореолы.
        // Применяется ПЕРВЫМ в post-цепочке, до post_sharp.
        bool  dehalo = false;
        float dehalo_strength = 0.5f;
        float dehalo_dark_bias = 0.3f;

        bool  post_sharp = false;
        float post_sharp_amount = 0.30f;
        float post_sharp_halo_guard = 0.06f;
    };

    // Извлечь размеры кадра и fps через ffprobe. Возвращает false при ошибке.
    bool probe_video(const std::string& path, int& w, int& h, double& fps, bool& has_audio);

    // Обработать видео: 2× через mvmod10.
    // out_w / out_h = 0 → авто (= 2× вход). Иначе — целевой размер (с post-resize, если надо).
    bool process_video(const std::string& in_path,
        const std::string& out_path,
        const std::string& model_path,
        int out_w, int out_h,
        const VideoOptions& opts = {});

    // Обработать одно изображение (PNG/JPG) через mvmod10.
    bool process_image(const std::string& in_path,
        const std::string& out_path,
        const std::string& model_path,
        int out_w, int out_h,
        const VideoOptions& opts = {});

}