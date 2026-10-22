#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "mvmod10.h"
#include "mvmod10_filters.h"
#include "mvmod10_engine.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>

namespace mvmod10 {

#ifdef _WIN32

    // ===================== Child process с pipes =====================

    struct ChildProc {
        HANDLE hProcess = NULL;
        HANDLE hReadChild = NULL;    // если мы читаем из stdout дочернего
        HANDLE hWriteChild = NULL;   // если мы пишем в stdin дочернего

        bool start(const std::string& cmd, bool want_read, bool want_write) {
            SECURITY_ATTRIBUTES sa = {};
            sa.nLength = sizeof(sa);
            sa.bInheritHandle = TRUE;

            HANDLE hChildOutW = NULL, hChildInR = NULL;

            if (want_read) {
                if (!CreatePipe(&hReadChild, &hChildOutW, &sa, 1 << 22)) return false;
                SetHandleInformation(hReadChild, HANDLE_FLAG_INHERIT, 0);
            }
            if (want_write) {
                if (!CreatePipe(&hChildInR, &hWriteChild, &sa, 1 << 22)) return false;
                SetHandleInformation(hWriteChild, HANDLE_FLAG_INHERIT, 0);
            }

            STARTUPINFOA si = {};
            si.cb = sizeof(si);
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdInput = want_write ? hChildInR : GetStdHandle(STD_INPUT_HANDLE);
            si.hStdOutput = want_read ? hChildOutW : GetStdHandle(STD_OUTPUT_HANDLE);
            si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

            PROCESS_INFORMATION pi = {};
            std::vector<char> buf(cmd.begin(), cmd.end());
            buf.push_back(0);

            BOOL ok = CreateProcessA(NULL, buf.data(), NULL, NULL, TRUE,
                0, NULL, NULL, &si, &pi);
            if (want_read && hChildOutW) CloseHandle(hChildOutW);
            if (want_write && hChildInR)  CloseHandle(hChildInR);
            if (!ok) return false;

            hProcess = pi.hProcess;
            CloseHandle(pi.hThread);
            return true;
        }

        DWORD read(void* buf, DWORD n) {
            DWORD r = 0;
            if (!hReadChild) return 0;
            ReadFile(hReadChild, buf, n, &r, NULL);
            return r;
        }
        DWORD write(const void* buf, DWORD n) {
            DWORD w = 0;
            if (!hWriteChild) return 0;
            WriteFile(hWriteChild, buf, n, &w, NULL);
            return w;
        }
        void close() {
            if (hWriteChild) { CloseHandle(hWriteChild); hWriteChild = NULL; }
            if (hReadChild) { CloseHandle(hReadChild);  hReadChild = NULL; }
            if (hProcess) {
                WaitForSingleObject(hProcess, INFINITE);
                CloseHandle(hProcess); hProcess = NULL;
            }
        }
        ~ChildProc() { close(); }
    };

    // ===================== утилиты ffmpeg =====================

    static bool run_capture(const std::string& cmd, std::string& out) {
        out.clear();
        FILE* p = _popen(cmd.c_str(), "r");
        if (!p) return false;
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), p)) > 0)
            out.append(buf, n);
        int rc = _pclose(p);
        return rc == 0;
    }

    bool probe_video(const std::string& path, int& w, int& h, double& fps, bool& has_audio) {
        w = h = 0; fps = 0.0; has_audio = false;

        // Используем avg_frame_rate вместо r_frame_rate.
        // Для interlaced MPEG-1/MPEG-2 r_frame_rate = field rate (59.94),
        // а avg_frame_rate = реальный frame rate (29.97). Именно avg нужен.
        std::string cmd =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=width,height,avg_frame_rate,r_frame_rate "
            "-of csv=p=0 \"" + path + "\"";
        std::string out;
        if (!run_capture(cmd, out)) return false;
        // Формат: "W,H,avg_rate,r_rate\n"  (любой из rate может быть '0/0')
        int a = 0, b = 0;
        char avg_num[32] = { 0 }, avg_den[32] = { 0 };
        char r_num[32] = { 0 }, r_den[32] = { 0 };

        if (std::sscanf(out.c_str(), "%d,%d,%31[0-9]/%31[0-9],%31[0-9]/%31[0-9]",
            &a, &b, avg_num, avg_den, r_num, r_den) == 6) {
            w = a; h = b;
            double an = std::atof(avg_num), ad = std::atof(avg_den);
            double rn = std::atof(r_num), rd = std::atof(r_den);
            if (an > 0.0 && ad > 0.0) {
                fps = an / ad;                 // предпочитаем avg
            }
            else if (rn > 0.0 && rd > 0.0) {
                fps = rn / rd;                 // fallback на r
            }
            else {
                fps = 25.0;
            }
        }
        else {
            // Fallback — старый парсинг на случай, если avg/r пришли по-другому
            char num[64] = { 0 }, den[64] = { 0 };
            if (std::sscanf(out.c_str(), "%d,%d,%63[0-9]/%63[0-9]",
                &a, &b, num, den) == 4) {
                w = a; h = b;
                double n = std::atof(num), d = std::atof(den);
                fps = (d > 0) ? n / d : 25.0;
            }
            else {
                return false;
            }
        }

        // ---- Детект interlaced. Для field-coded источников (MPEG-1/2)
        // ffprobe возвращает field_rate (удвоенный frame_rate). Мы используем
        // frame_rate, поэтому при interlaced делим fps на 2.
        std::string cmd_fo =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=field_order -of csv=p=0 \"" + path + "\"";
        std::string fo;
        if (run_capture(cmd_fo, fo)) {
            while (!fo.empty() && (fo.back() == '\n' || fo.back() == '\r'
                || fo.back() == ' '))
                fo.pop_back();
            std::cerr << "[mvmod10] field_order=" << fo << " fps_raw=" << fps << "\n";
            if (fo == "tt" || fo == "bb" || fo == "tb" || fo == "bt") {
                fps = fps / 2.0;
                std::cerr << "[mvmod10] interlaced detected -> fps=" << fps << "\n";
            }
        }

        std::string cmd2 =
            "ffprobe -v error -select_streams a -show_entries stream=index "
            "-of csv=p=0 \"" + path + "\"";
        std::string out2;
        if (run_capture(cmd2, out2) && !out2.empty()) has_audio = true;
        return true;
    }

    static void print_progress(long long frame, long long total,
        double elapsed, const char* prefix = "");
    static long long probe_frame_count(const std::string& path, double fps);

    // ===================== обработка видео =====================

    bool process_video(const std::string& in_path,
        const std::string& out_path,
        const std::string& model_path,
        int out_w, int out_h,
        const VideoOptions& opts)
    {
        int w_in, h_in; double fps; bool has_a;
        if (!probe_video(in_path, w_in, h_in, fps, has_a)) {
            std::cerr << "[mvmod10] ffprobe failed on: " << in_path << "\n";
            return false;
        }
        if (opts.fps_override > 0.0) {
            std::cerr << "[mvmod10] fps override: " << fps
                << " -> " << opts.fps_override << "\n";
            fps = opts.fps_override;
        }
        if (out_w <= 0) out_w = w_in * 2;
        if (out_h <= 0) out_h = h_in * 2;

        std::cout << "[mvmod10] source: " << w_in << "x" << h_in
            << " @ " << fps << " fps | audio: " << (has_a ? "yes" : "no") << "\n";
        std::cout << "[mvmod10] output: " << out_w << "x" << out_h << "\n";
        std::cout << "[mvmod10] loading model: " << model_path << "\n";

        MVNNModel model;
        if (!model.load(model_path)) {
            std::cerr << "[mvmod10] cannot load MVNN model\n";
            return false;
        }
        std::cout << "[mvmod10] model: feat=" << model.feat
            << " tile=" << model.tile
            << " e=" << model.e0_blocks << "/" << model.e1_blocks << "/" << model.e2_blocks
            << " scale=" << model.default_scale << "\n";

        long long total_frames = probe_frame_count(in_path, fps);
        if (total_frames > 0)
            std::cout << "[mvmod10] total frames: ~" << total_frames << "\n";
        else
            std::cout << "[mvmod10] total frames: unknown\n";

        std::cout << "[mvmod10] starting ffmpeg subprocesses..." << std::endl;

        // yadif=0:-1:0 — деинтерлейс (для field-coded MPEG-1/2 объединяет
        // пары полей в один кадр). Для progressive источников — no-op.
        // НЕ указываем -r и -fps_mode: yadif сам выставит правильный fps,
        // а ffmpeg по умолчанию отдаст CFR без дублирования кадров.
        std::ostringstream cin_cmd;
        cin_cmd << "ffmpeg -v warning -i \"" << in_path << "\" "
            << "-vf yadif=0:-1:0 "
            << "-f rawvideo -pix_fmt rgb24 "
            << "-";
        std::cerr << "[mvmod10] ffmpeg in: " << cin_cmd.str() << "\n";

        ChildProc fin;
        if (!fin.start(cin_cmd.str(), /*read=*/true, /*write=*/false)) {
            std::cerr << "[mvmod10] cannot start input ffmpeg\n";
            return false;
        }

        // Однопроходный вариант — проверен, работает с аудио и видео.
        // (Откат после экспериментов с двухпроходным режимом.)
        std::ostringstream cout_cmd;
        cout_cmd << "ffmpeg -y -v warning -f image2pipe -vcodec ppm "
            << "-framerate " << fps << " -i - ";
        if (has_a)
            cout_cmd << "-i \"" << in_path << "\" "
            << "-map 0:v -map 1:a? -c:a copy ";
        cout_cmd << "-vf format=yuv420p -c:v libx264 -preset veryfast "
            << "-crf 17 -r " << fps << " "
            << "\"" << out_path << "\"";

        std::cerr << "[mvmod10] ffmpeg out: " << cout_cmd.str() << "\n";

        ChildProc fout;
        if (!fout.start(cout_cmd.str(), /*read=*/false, /*write=*/true)) {
            std::cerr << "[mvmod10] cannot start output ffmpeg\n";
            return false;
        }

        // Размер кадра во входе
        size_t in_frame_bytes = (size_t)w_in * h_in * 3;
        std::vector<uint8_t> in_buf(in_frame_bytes);
        std::vector<uint8_t> out_buf((size_t)out_w * out_h * 3);

        Tensor state;
        Tensor prev_lr;
        bool have_prev = false;
        int frame_idx = 0;
        auto t0 = std::chrono::steady_clock::now();

        FilterState fstate;
        Tensor lr_filtered, out_filtered;

        FilterOptions fopt;
        fopt.temporal_denoise = opts.temporal_denoise;
        fopt.denoise_strength = opts.denoise_strength;
        fopt.denoise_motion_thresh = opts.denoise_motion_thresh;
        fopt.pre_sharp = opts.pre_sharp;
        fopt.pre_sharp_amount = opts.pre_sharp_amount;
        fopt.post_sharp = opts.post_sharp;
        fopt.post_sharp_amount = opts.post_sharp_amount;
        fopt.post_sharp_halo_guard = opts.post_sharp_halo_guard;
        fopt.dehalo = opts.dehalo;
        fopt.dehalo_strength = opts.dehalo_strength;
        fopt.dehalo_dark_bias = opts.dehalo_dark_bias;

        while (true) {
            DWORD got = 0;
            size_t total = 0;
            while (total < in_frame_bytes) {
                DWORD chunk = fin.read(in_buf.data() + total,
                    (DWORD)(in_frame_bytes - total));
                if (chunk == 0) break;
                total += chunk;
            }
            if (total != in_frame_bytes) break;

            Tensor lr = uint8_hwc_to_tensor(in_buf.data(), h_in, w_in);

            // Scene cut detection
            bool scene_cut = false;
            if (have_prev && opts.stateful) {
                float diff = 0.f;
                int HW = w_in * h_in;
                for (int i = 0; i < HW; ++i)
                    for (int c = 0; c < 3; ++c)
                        diff += std::fabs(lr.data[c * HW + i] - prev_lr.data[c * HW + i]);
                diff /= (float)(HW * 3);
                if (diff > opts.scene_cut_thresh) {
                    state = Tensor();  // clear
                    fstate.reset();
                    scene_cut = true;
                }
            }

            // Pre-filter
            apply_pre_filter(lr, lr_filtered, fstate, fopt);

            // Модель работает на отфильтрованном входе
            Tensor next_state;
            Tensor out = model.forward(
                lr_filtered,
                have_prev ? &prev_lr : nullptr,     // prev_lr = предыдущий СЫРОЙ (для motion)
                (opts.stateful && state.numel() > 0) ? &state : nullptr,
                (opts.stateful) ? &next_state : nullptr,
                model.default_scale,
                opts.strength,
                /*clamp_out=*/true
            );

            // Post-filter
            apply_post_filter(out, out_filtered, fstate, fopt);

            if (opts.stateful) state = std::move(next_state);
            prev_lr = lr;  // обновляем до СЫРОГО кадра
            have_prev = true;

            // Если запрошен другой масштаб — ресайзим (bicubic) внутри ffmpeg через PPM?
            // Пока поддерживаем только 2×. Если out_w != 2*w_in — вернём ошибку.
            if (out_w != w_in * 2 || out_h != h_in * 2) {
                std::cerr << "[mvmod10] only 2x output supported for now\n";
                return false;
            }

            tensor_to_uint8_hwc(out_filtered, out_buf.data());

            // PPM header + payload
            char hdr[64];
            int hdr_len = std::snprintf(hdr, sizeof(hdr),
                "P6\n%d %d\n255\n", out_w, out_h);
            fout.write(hdr, hdr_len);
            size_t total_written = 0;
            while (total_written < out_buf.size()) {
                DWORD w = fout.write(out_buf.data() + total_written,
                    (DWORD)(out_buf.size() - total_written));
                if (w == 0) break;
                total_written += w;
            }

            ++frame_idx;

            if (opts.verbose) {
                if (frame_idx == 1) {
                    auto now = std::chrono::steady_clock::now();
                    double el = std::chrono::duration<double>(now - t0).count();
                    std::cout << "\n[mvmod10] first frame done in " << el
                        << "s (model speed: " << (1.0 / max(el, 1e-6))
                        << " fps)\n";
                }
                // Обновляем прогресс-бар: каждые 5 кадров ИЛИ раз в 2 секунды
                if (frame_idx % 5 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    double el = std::chrono::duration<double>(now - t0).count();
                    print_progress(frame_idx, total_frames, el);
                }
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(t_end - t0).count();
        std::cout << "\n[mvmod10] done: " << frame_idx << " frames in "
            << el << "s (" << (frame_idx / max(el, 1e-6)) << " fps)\n";
        std::cout << "[mvmod10] file: " << out_path << "\n";

        fin.close();
        fout.close();
        return true;
    }

    // ===================== прогресс-бар =====================

    static void print_progress(long long frame, long long total,
        double elapsed, const char* prefix)
    {
        double fps = frame / max(elapsed, 1e-6);

        if (total > 0) {
            double frac = (double)frame / (double)total;
            if (frac > 1.0) frac = 1.0;

            const int bar_w = 28;
            int filled = (int)(frac * bar_w);

            char bar[64];
            for (int i = 0; i < bar_w; ++i) bar[i] = (i < filled) ? '=' : ' ';
            bar[bar_w] = 0;

            double eta = (fps > 0.1) ? (total - frame) / fps : 0.0;
            int eta_m = (int)(eta / 60.0);
            int eta_s = (int)eta - eta_m * 60;

            std::printf("\r[mvmod10]%s [%s] %lld/%lld  %.2f fps  ETA %02d:%02d    ",
                prefix, bar, frame, total, fps, eta_m, eta_s);
        }
        else {
            std::printf("\r[mvmod10]%s %lld frames  %.2f fps  %.1fs    ",
                prefix, frame, fps, elapsed);
        }
        std::fflush(stdout);
    }

    // Оценка общего количества кадров. Возвращает 0 при неудаче.
    static long long probe_frame_count(const std::string& path, double fps)
    {
        std::string out;

        // 1) Пробуем через nb_frames (для mp4/mkv обычно есть)
        std::string cmd1 =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=nb_frames -of csv=p=0 \"" + path + "\"";
        if (run_capture(cmd1, out)) {
            long long n = std::atoll(out.c_str());
            if (n > 0) return n;
        }

        // 2) Fallback — duration * fps
        std::string cmd2 =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries format=duration -of csv=p=0 \"" + path + "\"";
        if (run_capture(cmd2, out)) {
            double dur = std::atof(out.c_str());
            if (dur > 0.0) return (long long)(dur * fps + 0.5);
        }

        return 0;
    }

    bool process_image(const std::string& in_path,
        const std::string& out_path,
        const std::string& model_path,
        int out_w, int out_h,
        const VideoOptions& opts)
    {
        // Для одиночного изображения используем PNG-путь через PIL-подобный код.
        // Простейший путь — обернуть в однокадровое видео через ffmpeg.
        // Временный минимум: рапортуем, что не поддерживается.
        std::cerr << "[mvmod10] process_image not implemented in C++ yet — "
            "используйте PyTorch inference2.py для тестов PNG\n";
        return false;
    }

#else  // не Windows

    bool probe_video(const std::string&, int&, int&, double&, bool&) { return false; }
    bool process_video(const std::string&, const std::string&, const std::string&,
        int, int, const VideoOptions&) {
        return false;
    }
    bool process_image(const std::string&, const std::string&, const std::string&,
        int, int, const VideoOptions&) {
        return false;
    }

#endif

}