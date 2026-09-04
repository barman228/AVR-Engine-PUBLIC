#define NOMINMAX

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <deque>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>

// NCNN
#include "ncnn/net.h"
#include "ncnn/mat.h"

// FFmpeg
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/error.h>
}

bool run_video_upscaler(const std::string& model_prefix, const std::string& input_path, const std::string& output_path,
    int target_w = 0, int target_h = 0) {

    std::cout << "[DEBUG] Инициализация NCNN сети...\n";
    ncnn::Net net;
    net.opt.num_threads = 0;
    net.opt.use_vulkan_compute = false;

    std::string param_path = model_prefix + ".ncnn.param";
    std::string bin_path = model_prefix + ".ncnn.bin";
    std::cout << "[DEBUG] Загрузка модели: " << param_path << " / " << bin_path << "\n";

    if (net.load_param(param_path.c_str()) != 0) {
        std::cerr << "[ОШИБКА] Не удалось загрузить .param файл модели: " << param_path << "\n";
        return false;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        std::cerr << "[ОШИБКА] Не удалось загрузить .bin файл модели: " << bin_path << "\n";
        return false;
    }

    std::cout << "[DEBUG] Открытие входного видеофайла: " << input_path << "\n";
    AVFormatContext* in_ctx = nullptr;
    if (avformat_open_input(&in_ctx, input_path.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "[ОШИБКА] Не удалось открыть входной файл: " << input_path << "\n";
        return false;
    }

    std::cout << "[DEBUG] Поиск информации о потоках...\n";
    if (avformat_find_stream_info(in_ctx, nullptr) < 0) {
        std::cerr << "[ОШИБКА] Не удалось найти информацию о потоках в файле.\n";
        avformat_close_input(&in_ctx);
        return false;
    }

    int v_idx = -1, a_idx = -1;
    for (unsigned int i = 0; i < in_ctx->nb_streams; i++) {
        if (in_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && v_idx == -1) v_idx = i;
        else if (in_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && a_idx == -1) a_idx = i;
    }

    std::cout << "[DEBUG] Индексы потоков -> Видео: " << v_idx << ", Аудио: " << a_idx << "\n";
    if (v_idx == -1) {
        std::cerr << "[ОШИБКА] Видеопоток не найден в файле.\n";
        avformat_close_input(&in_ctx);
        return false;
    }

    AVStream* in_stream = in_ctx->streams[v_idx];
    AVStream* in_audio_stream = (a_idx != -1) ? in_ctx->streams[a_idx] : nullptr;

    std::cout << "[DEBUG] Инициализация декодера...\n";
    const AVCodec* decoder = avcodec_find_decoder(in_stream->codecpar->codec_id);
    if (!decoder) {
        std::cerr << "[ОШИБКА] Декодер для видео не найден.\n";
        avformat_close_input(&in_ctx);
        return false;
    }

    AVCodecContext* dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, in_stream->codecpar);
    if (avcodec_open2(dec_ctx, decoder, nullptr) < 0) {
        std::cerr << "[ОШИБКА] Не удалось открыть декодер.\n";
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_ctx);
        return false;
    }

    int in_w = dec_ctx->width;
    int in_h = dec_ctx->height;
    std::cout << "[DEBUG] Разрешение входного видео: " << in_w << "x" << in_h << "\n";

    const int nn_scale = 2;
    const int num_frames = 2;
    const float qp_value = 0.5f;

    const char* in_frames_blob = "in0";
    const char* in_qp_blob = "in1";
    const char* out_blob = "out0";

    int nn_w = in_w * nn_scale;
    int nn_h = in_h * nn_scale;

    if (target_w <= 0) target_w = nn_w;
    if (target_h <= 0) target_h = nn_h;
    std::cout << "[DEBUG] Целевое разрешение выхода: " << target_w << "x" << target_h << "\n";

    std::cout << "[DEBUG] Инициализация кодера MJPEG...\n";
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, nullptr, output_path.c_str());
    if (!out_ctx) {
        std::cerr << "[ОШИБКА] Не удалось создать выходной контекст для: " << output_path << "\n";
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_ctx);
        return false;
    }

    const AVCodec* encoder = avcodec_find_encoder_by_name("mjpeg");
    if (!encoder) encoder = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!encoder) {
        std::cerr << "[ОШИБКА] Энкодер MJPEG не найден в системе.\n";
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&in_ctx);
        avformat_free_context(out_ctx);
        return false;
    }

    AVStream* out_stream = avformat_new_stream(out_ctx, encoder);
    AVCodecContext* enc_ctx = avcodec_alloc_context3(encoder);

    enc_ctx->width = target_w;
    enc_ctx->height = target_h;
    enc_ctx->time_base = in_stream->time_base;

    AVRational framerate = in_stream->r_frame_rate;
    if (framerate.num <= 0 || framerate.den <= 0) {
        framerate = av_guess_frame_rate(in_ctx, in_stream, nullptr);
    }
    if (framerate.num <= 0 || framerate.den <= 0) {
        framerate = { 30, 1 };
    }
    enc_ctx->framerate = framerate;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (out_ctx->oformat->flags & AVFMT_GLOBALHEADER) enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    AVDictionary* opt = nullptr;
    av_dict_set(&opt, "q:v", "2", 0);

    if (avcodec_open2(enc_ctx, encoder, &opt) < 0) {
        std::cerr << "[ОШИБКА] Не удалось открыть энкодер MJPEG.\n";
        av_dict_free(&opt);
        avcodec_free_context(&dec_ctx);
        avcodec_free_context(&enc_ctx);
        avformat_close_input(&in_ctx);
        avformat_free_context(out_ctx);
        return false;
    }
    av_dict_free(&opt);
    avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);

    AVStream* out_audio_stream = nullptr;
    if (in_audio_stream) {
        out_audio_stream = avformat_new_stream(out_ctx, nullptr);
        avcodec_parameters_copy(out_audio_stream->codecpar, in_audio_stream->codecpar);
        out_audio_stream->codecpar->codec_tag = 0;
        out_audio_stream->time_base = in_audio_stream->time_base;
    }

    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        std::cout << "[DEBUG] Открытие файла для записи: " << output_path << "\n";
        if (avio_open(&out_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "[ОШИБКА] Не удалось открыть выходной файл для записи: " << output_path << "\n";
            avcodec_free_context(&dec_ctx);
            avcodec_free_context(&enc_ctx);
            avformat_close_input(&in_ctx);
            avformat_free_context(out_ctx);
            return false;
        }
    }

    std::cout << "[DEBUG] Запись заголовка видеофайла...\n";
    if (avformat_write_header(out_ctx, nullptr) < 0) {
        std::cerr << "[ОШИБКА] Не удалось записать заголовок (avformat_write_header).\n";
        // Очистка
        avcodec_free_context(&dec_ctx);
        avcodec_free_context(&enc_ctx);
        avformat_close_input(&in_ctx);
        avformat_free_context(out_ctx);
        return false;
    }
    avio_flush(out_ctx->pb);

    std::cout << "[DEBUG] Инициализация контекстов масшатбирования (SwsContext)...\n";
    SwsContext* sws_to_8bit = sws_getContext(in_w, in_h, dec_ctx->pix_fmt, in_w, in_h, AV_PIX_FMT_YUV420P, SWS_POINT, nullptr, nullptr, nullptr);
    SwsContext* sws_chroma = sws_getContext(in_w, in_h, dec_ctx->pix_fmt, target_w, target_h, AV_PIX_FMT_YUV420P, SWS_LANCZOS, nullptr, nullptr, nullptr);
    SwsContext* sws_y_out = nullptr;
    if (target_w != nn_w || target_h != nn_h) {
        sws_y_out = sws_getContext(nn_w, nn_h, AV_PIX_FMT_GRAY8, target_w, target_h, AV_PIX_FMT_GRAY8, SWS_LANCZOS, nullptr, nullptr, nullptr);
    }

    std::cout << "[DEBUG] Выделение памяти под кадры...\n";
    AVFrame* frame = av_frame_alloc();
    AVFrame* frame_8bit = av_frame_alloc();
    frame_8bit->format = AV_PIX_FMT_YUV420P;
    frame_8bit->width = in_w;
    frame_8bit->height = in_h;
    av_frame_get_buffer(frame_8bit, 32);

    AVFrame* out_frame = av_frame_alloc();
    out_frame->format = enc_ctx->pix_fmt;
    out_frame->width = target_w;
    out_frame->height = target_h;
    av_frame_get_buffer(out_frame, 32);

    std::deque<std::vector<uint8_t>> y_history;

    int64_t total_frames = in_stream->nb_frames;
    if (total_frames <= 0 && in_stream->duration > 0) {
        total_frames = static_cast<int64_t>(in_stream->duration * av_q2d(in_stream->time_base) * av_q2d(framerate));
    }

    std::cout << "[DEBUG] Запуск главного цикла обработки кадров (всего кадров: " << total_frames << ")...\n";
    int frame_count = 0;
    AVPacket* pkt = av_packet_alloc();
    std::queue<AVPacket*> audio_queue;

    std::vector<float> norm_vals_in(num_frames, 1.0f / 255.0f);
    std::vector<float> mean_vals_in(num_frames, 0.0f);
    std::vector<float> norm_vals_out(1, 255.0f);
    std::vector<float> mean_vals_out(1, 0.0f);

    while (av_read_frame(in_ctx, pkt) >= 0) {
        if (a_idx != -1 && pkt->stream_index == a_idx) {
            AVPacket* cloned_pkt = av_packet_clone(pkt);
            cloned_pkt->stream_index = out_audio_stream->index;
            av_packet_rescale_ts(cloned_pkt, in_audio_stream->time_base, out_audio_stream->time_base);
            audio_queue.push(cloned_pkt);
            av_packet_unref(pkt);
            continue;
        }

        if (pkt->stream_index == v_idx) {
            avcodec_send_packet(dec_ctx, pkt);
            while (avcodec_receive_frame(dec_ctx, frame) == 0) {

                av_frame_make_writable(out_frame);

                sws_scale(sws_to_8bit, frame->data, frame->linesize, 0, in_h, frame_8bit->data, frame_8bit->linesize);

                std::vector<uint8_t> y_plane(in_w * in_h);
                for (int y = 0; y < in_h; ++y) {
                    std::memcpy(y_plane.data() + y * in_w, frame_8bit->data[0] + y * frame_8bit->linesize[0], in_w);
                }

                if (y_history.empty()) {
                    for (int i = 0; i < num_frames - 1; ++i) {
                        y_history.push_back(y_plane);
                    }
                }
                y_history.push_back(y_plane);
                if (y_history.size() > num_frames) {
                    y_history.pop_front();
                }

                sws_scale(sws_chroma, frame->data, frame->linesize, 0, in_h, out_frame->data, out_frame->linesize);

                ncnn::Mat multi_frame_mat(in_w, in_h, num_frames);
                for (int c = 0; c < num_frames; ++c) {
                    ncnn::Mat channel_mat = ncnn::Mat::from_pixels(
                        y_history[c].data(), ncnn::Mat::PIXEL_GRAY, in_w, in_h
                    );
                    std::memcpy(multi_frame_mat.channel(c), channel_mat.data, in_w * in_h * sizeof(float));
                }

                multi_frame_mat.substract_mean_normalize(mean_vals_in.data(), norm_vals_in.data());

                ncnn::Mat qp_mat(1);
                qp_mat[0] = qp_value;

                ncnn::Extractor ex = net.create_extractor();
                ex.input(in_frames_blob, multi_frame_mat);
                ex.input(in_qp_blob, qp_mat);

                ncnn::Mat out_patch;
                if (ex.extract(out_blob, out_patch) != 0) {
                    std::cerr << "\n[ОШИБКА] Сбой инференса на кадре " << frame_count << "\n";
                    continue;
                }

                out_patch.substract_mean_normalize(mean_vals_out.data(), norm_vals_out.data());

                std::vector<uint8_t> nn_y_plane(nn_w * nn_h);
                out_patch.to_pixels(nn_y_plane.data(), ncnn::Mat::PIXEL_GRAY, nn_w);

                if (target_w == nn_w && target_h == nn_h) {
                    for (int y = 0; y < nn_h; ++y) {
                        std::memcpy(out_frame->data[0] + y * out_frame->linesize[0],
                            nn_y_plane.data() + y * nn_w, nn_w);
                    }
                }
                else {
                    const uint8_t* in_data[1] = { nn_y_plane.data() };
                    int in_linesize[1] = { nn_w };
                    sws_scale(sws_y_out, in_data, in_linesize, 0, nn_h,
                        out_frame->data, out_frame->linesize);
                }

                out_frame->pts = frame->pts;

                avcodec_send_frame(enc_ctx, out_frame);
                AVPacket* out_pkt = av_packet_alloc();
                while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                    out_pkt->stream_index = out_stream->index;
                    av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);

                    while (!audio_queue.empty()) {
                        AVPacket* a_pkt = audio_queue.front();
                        if (av_rescale_q(a_pkt->pts, out_audio_stream->time_base, in_stream->time_base) <= frame->pts) {
                            av_interleaved_write_frame(out_ctx, a_pkt);
                            avio_flush(out_ctx->pb);
                            av_packet_free(&a_pkt);
                            audio_queue.pop();
                        }
                        else break;
                    }

                    av_interleaved_write_frame(out_ctx, out_pkt);
                    avio_flush(out_ctx->pb);

                    av_packet_unref(out_pkt);
                }
                av_packet_free(&out_pkt);

                ++frame_count;
                if (total_frames > 0) {
                    std::cout << "\rОбработано кадров: [" << frame_count << "/" << total_frames << "]   " << std::flush;
                }
            }
        }
        av_packet_unref(pkt);
    }

    avcodec_send_frame(enc_ctx, nullptr);
    while (avcodec_receive_packet(enc_ctx, pkt) == 0) {
        pkt->stream_index = out_stream->index;
        av_packet_rescale_ts(pkt, enc_ctx->time_base, out_stream->time_base);
        av_interleaved_write_frame(out_ctx, pkt);
        avio_flush(out_ctx->pb);
        av_packet_unref(pkt);
    }

    while (!audio_queue.empty()) {
        AVPacket* a_pkt = audio_queue.front();
        av_interleaved_write_frame(out_ctx, a_pkt);
        avio_flush(out_ctx->pb);
        av_packet_free(&a_pkt);
        audio_queue.pop();
    }

    av_write_trailer(out_ctx);
    avio_flush(out_ctx->pb);

    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&frame_8bit);
    av_frame_free(&out_frame);
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    avformat_close_input(&in_ctx);
    avformat_free_context(out_ctx);
    sws_freeContext(sws_to_8bit);
    sws_freeContext(sws_chroma);
    if (sws_y_out) sws_freeContext(sws_y_out);

    std::cout << "\n[INFO] Успешно завершено!\n";
    return true;
}