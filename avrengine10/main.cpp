/*
HFNAR and AVREngine 2026 by Gabella Studios Network
Writer and Idea: Artem Mikhailov

HFNAR (High Frequency Noise Audio Reconstructor)  -
is a software tool designed to enhance audio quality by reconstructing high-frequency noise components. 
It supports multiple versions, each with different capabilities and features. 
The software can process audio files in WAV format, allowing users to specify various parameters such as version, 
cutoff frequency, and model files for neural network-based processing.

HFNAR Created specially for LMG Archive project by GSN 2023-2026.
*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include "detector.h"
#include "hfnar10.h"
#include "hfnar20.h"
#include "hfnar21.h"
#include "hfnar30.h"
#include "hfnar32.h"
#include "hfnar34.h"
#include "hfnar40p.h"
#include "hfnar40a.h"
#include "hfnar40.h"
#include "thread"
//#include "mvmod10_nn.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int main(int argc, char* argv[]) {
    std::string in_filename = "";
    std::string out_filename = "";
    int version = 32;             // версия по дефолту (HFNAR v2.0)
    double manual_cutoff = -1.0;  // Меньше 0 означает, что работает автодетектор
	bool TES = false; // Adaptive Temporal Envelope Sharping (TES) флаг для HFNAR32
    bool force_mono_flag = false;
    std::string model_filename = "standart_taka.vonn"; // Дефолтная непараметрическая модель .mv32 (vomod 2.0)
	std::string model_filename_mv1x = "MVMOD/video_enhancer"; // Дефолтная видео модель .mv1x (vomod 2.4)
	std::string model_filename_vnn4 = "vomod40_model.vnn4"; // Дефолтная модель VNN4 для HFNAR40 (vomod 4.0)
	int w = 0; int h = 0; int num_threads = 0; // Количество потоков по умолчанию

    // (флаги можно писать в любом порядке)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-v") {
            if (i + 1 < argc) {
                version = std::atoi(argv[++i]);
                if (version != 10 && version != 16 && version != 20 && version != 21 && version != 30 && version != 32 && version != 34 && version != 40 && version != 41 && version != 42) {
                    std::cerr << "Error: version not found, use 10,16,20,21,30,32.\n";
                    return 1;
                }
            }
            else {
                std::cerr << "Error: Flag -v must be specified (10,16,20,21,30,32).\n";
                return 1;
            }
        }
        else if (arg == "-genmod") {
            if (i + 1 < argc) {
                in_filename = argv[++i];
                unsigned int channels = 0;
                unsigned int sampleRate = 0;
                unsigned long long totalPCMFrameCount = 0;

                // 1. Чтение входного WAV-файла (донор для генерации ИЛИ низкочастотный исходник для апскейла)
                float* pInputData = drwav_open_file_and_read_pcm_frames_f32(in_filename.c_str(), &channels, &sampleRate, &totalPCMFrameCount, NULL);
                if (!pInputData) {
                    std::cerr << "Error: Cannot open or read input WAV file: " << in_filename << "\n";
                    return 1;
                }
                std::cout << "[Mode] Generation of voice fingerprint model from high-quality reference...\n";
                std::cout << "Reference file: " << in_filename << "\n";
                std::cout << "Target model path: " << model_filename << "\n";

                bool success = HFNAR32::generate_voice_model32(pInputData, totalPCMFrameCount, channels, sampleRate, model_filename);

                drwav_free(pInputData, NULL);
                if (success) {
                    std::cout << "[Success] Model saved to " << model_filename << "\n";
                    return 0;
                }
                else {
                    std::cerr << "[Error] Failed to generate voice model.\n";
                    return 1;
                }
			}
            else {
                std::cerr << "Error: Flag -genmod must be followed by a reference WAV file path.\n";
                return 1;
            }
        }
        else if (arg == "-cutoff") {
            if (i + 1 < argc) {
                manual_cutoff = std::stod(argv[++i]);
                if (manual_cutoff <= 3600.0) {
                    std::cerr << "Error: Cut-off frequency need to be more than 3600.1Hz.\n";
                    return 1;
                }
            }
            else {
                std::cerr << "Error: flag -cutoff not valid.\n";
                return 1;
            }
		}
		else if (arg == "-tes") {
			TES = true; std::cerr << "TES - Enabled!\n";
		}
		else if (arg == "-threads") {
			if (i + 1 < argc) {
				num_threads = std::stoi(argv[++i]);
				if (num_threads < 0) {
					std::cerr << "Error: Number of threads must be a positive integer.\n";
					return 1;
				}
			}
			else {
				std::cerr << "Error: Flag -threads must be followed by a number.\n";
				return 1;
			}
		}
		else if (arg == "-model") {
			if (i + 1 < argc) {
				model_filename = argv[++i];
			}
			else {
				std::cerr << "Error: Flag -model must be followed by a model filename.\n";
				return 1;
			}
			if (version == 16) {
				model_filename_mv1x = model_filename; // Для версии 1.6 используем другой путь к модели
			}
			if (version == 40) {
				model_filename_vnn4 = model_filename; // Для версии 4.0 используем другой путь к модели
			}
		}
        else if (arg == "-mono") {
			force_mono_flag = true; std::cerr << "Mono force - Enabled!\n";
        }
        else if (arg == "-w") {
            w = std::stoi(argv[++i]);
        }
        else if (arg == "-h") {
            h = std::stoi(argv[++i]);
        }
        else {
            // детектор пути к файлу
            if (in_filename.empty()) {
                in_filename = arg;
            }
            else if (out_filename.empty()) {
                out_filename = arg;
            }
            else {
                std::cerr << "Error: arg invalid: '" << arg << "'\n";
                return 1;
            }
        }
    }

    // Главное меню
    if (in_filename.empty() || out_filename.empty()) {
        std::cout << "==================================================================\n";
        std::cout << " HFNAR (High Frequency Noise Audio Reconstructor) CLI\n";
        std::cout << " AVREngine 1.1000 Series - Preview Code 14\n";
        std::cout << "==================================================================\n\n";
        std::cout << "Using: " << argv[0] << " <input_wav> <output_wav> [flags]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -v <10|16|20|21|30|32|34|40|41|42>        Version (Default: 32 - Highest version)\n";
        std::cout << "  -v 32                   - it's Beta and may cause troubles     \n";
        std::cout << "  -v 42                   - it's Demo and may cause troubles     \n\n";
        std::cout << "  -cutoff <Hz>         Manual Cut-off frequency in Hz (if not specified, sets auto)\n";
        std::cout << "  -tes                 Enable Temporal Envelope Sharping\n\n";
        std::cout << "  -model <filename>    Set model to use (if not specified, uses default)\n";
        std::cout << "Samples:\n";
        std::cout << "  " << argv[0] << " input.wav output.wav                     (v3.2 + auto)\n";
        std::cout << "  " << argv[0] << " input.wav -model low_profile.vonn output.wav  (v3.2 + lpmodel)\n";
        std::cout << "  " << argv[0] << " input.wav -cutoff 14000 output.wav       (v3.2 + Cut-off prefered on 14kHz)\n";
        return 1;
    }

    if (version == 16) {
        std::cout << "Input video file: " << in_filename << "\n";
        std::cout << "Start HFNAR v1.6 (videomode)... BLOCKED!\n";
        if (h == 0 || w == 0) { std::cout << "ERROR: Ident -w -h\n"; return 1; }
        //run_video_upscaler(model_filename_mv1x, in_filename, out_filename, w, h);
        return 0;
    }

    unsigned int channels;
    unsigned int sampleRate;
    drwav_uint64 totalPCMFrameCount;

    // Чтение исходного WAV файла
    float* pInputData = drwav_open_file_and_read_pcm_frames_f32(in_filename.c_str(), &channels, &sampleRate, &totalPCMFrameCount, NULL);
    if (pInputData == NULL) {
        std::cerr << "Error: File cannot be open or readed: " << in_filename << "\n";
        return 1;
    }

    std::cout << "Input file: " << in_filename << "\n";
    std::cout << "Options: " << sampleRate << " Hz, " << channels << " Channels\n";

    // Определение или назначение частоты среза
    std::vector<double> cutoffs;
    if (manual_cutoff > 0.0) {
        std::cout << "Mode: Manual frequency = " << manual_cutoff << " Hz. Detector disabled.\n";
        cutoffs.assign(channels, manual_cutoff);
    }
    else {
        std::cout << "Mode: Auto spectral audio analysis...\n";
        cutoffs = detect_cutoff_frequencies(pInputData, channels, sampleRate, totalPCMFrameCount);
        for (unsigned int ch = 0; ch < channels; ++ch) {
            std::cout << "  Channel " << ch << " | Auto-cutoff: " << cutoffs[ch] << " Hz\n";
        }
    }

    // Выделение памяти под выходной буфер
    std::vector<float> outputData(totalPCMFrameCount * channels, 0.0f);

    // Генерация высоких частот в зависимости от выбранной версии
    if (version == 10) {
        if (cutoffs[0] < 14000.0) {
            std::cout << "This version not officially support this Cut-off!\n";
            std::cout << "HFNARv1 uses this range (14000-18000)\n";
        }
        std::cout << "Start HFNAR v1 (Standart noise)...\n";
        upscale_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, cutoffs);
    }
    else if (version == 20) {
        if (cutoffs[0] < 12000.0) {
            std::cout << "This version not officially support this Cut-off!\n";
            std::cout << "HFNARv2 uses this range (12000-18000)\n";
        }
        std::cout << "Start HFNAR v2.0 (Spectral envelope copying)...\n";
        upscale2_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, cutoffs);
    }
    else if (version == 21) {
        if (cutoffs[0] < 12000.0) {
            std::cout << "This version not officially support this Cut-off!\n";
            std::cout << "HFNARv2.0 uses this range (12000-18000)\n";
        }
        std::cout << "Start HFNAR v2.1 (Spectral envelope copying + energy noise)...\n";
        upscale21_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, cutoffs);
    }
    else if (version == 30) {
        if (cutoffs[0] < 11000.0) {
            std::cout << "This version not officially support this Cut-off!\n";
            std::cout << "HFNAR3 uses this range (11000-18000)\n";
        }
        std::cout << "Start HFNAR v3.0 (Spectral envelope copying + energy noise)...\n";
        upscale30_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, cutoffs);
    }
    else if (version == 32) {
        std::cout << "Start HFNAR v3.2 (NN-based Predictor VOMOD30)...\n";
		if (cutoffs[0] < 6000.0 || cutoffs[0] > 14000.0) {
			std::cout << "No one model not officially support this Cut-off!\n";
			std::cout << "Official modelset uses this range (6000-14000)\n";
		}
		if ((cutoffs[0] < 8000.0 && cutoffs[0] >= 6000.0) && model_filename == "standart_taka.vonn") {
			model_filename = "low_profile.vonn"; std::cerr << "INFO: Auto-Switch to " << model_filename << "\n";
        }
        else if (cutoffs[0] < 6000.0 && (model_filename == "standart_taka.vonn" || model_filename == "low_profile.vonn")) {
            model_filename = "ultralow_profile.vonn"; std::cerr << "INFO: Auto-Switch to " << model_filename << "\n";
        }
        else if (cutoffs[0] >= 10000.0 && (model_filename == "low_profile.vonn" || model_filename == "standart_taka.vonn")) {
			model_filename = "conv_taka.vonn"; std::cerr << "INFO: Auto-Switch to " << model_filename << "\n";
        }
        std::ifstream f(model_filename.c_str());
        if (!f.good()) {
            std::cerr << "Error: Neural network model file not found: " << model_filename << "\n";
            std::cerr << "Please put a valid .vonn file next to the executable.\n";
            drwav_free(pInputData, NULL);
            return 1;
        }
        f.close();
        std::cout << "Using neural network model: " << model_filename << "\n";

        HFNAR32::upscale32_patch_audio(
            pInputData,
            outputData.data(),
            channels,
            sampleRate,
            totalPCMFrameCount,
            cutoffs,
            model_filename,
            TES,
			num_threads
        );
    }
    else if (version == 34) {
        std::cout << "Start HFNAR v3.4 (NN-based Stereo Predictor VOMOD30-pc4)...\n";
		if (model_filename == "standart_taka.vonn") {
			model_filename = "standart_stereo.vonn";
		}
        std::ifstream f(model_filename.c_str());
        if (!f.good()) {
            std::cerr << "Error: Neural network model file not found: " << model_filename << "\n";
            std::cerr << "Please put a valid .vonn file next to the executable.\n";
            drwav_free(pInputData, NULL);
            return 1;
        }
        f.close();
        std::cout << "Using neural network model: " << model_filename << "\n";
        HFNAR34::process_mono2stereo(
            pInputData,
            outputData.data(),
            channels,
            sampleRate,
            totalPCMFrameCount,
            model_filename,
            force_mono_flag
        );
    }
    else if (version == 40) {
        if (cutoffs[0] < 4000.0) {
            std::cout << "This version not officially support this Cut-off!\n";
            std::cout << "HFNAR V4.0 uses this range (4000-18000)\n";
        }
        std::cout << "HFNAR V4.0 is PREVIEW VERSION!\n";
        std::cout << "Start HFNAR v4.0p1 (Preview: Dynamic Spectral Hill & Extended Low-Cutoff Support)...\n";
        upscale40p1_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, cutoffs);
    }
    else if (version == 41) {
        if (cutoffs[0] < 4000.0) {
            std::cout << "This version not officially support this Cut-off!\n";
            std::cout << "HFNAR V4.0 uses this range (4000-18000)\n";
        }
        std::cout << "HFNAR V4.0 is PREVIEW VERSION!\n";
        std::cout << "Start HFNAR v4.0A (Preview: Dynamic Spectral Hill & Extended Low-Cutoff Support)...\n";
        upscale40a_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, cutoffs);
	}
	else if (version == 42) {
		if (cutoffs[0] < 4000.0) {
			std::cout << "This version not officially support this Cut-off!\n";
			std::cout << "HFNAR V4.2 uses this range (4000-18000)\n";
		}
		std::cout << "HFNAR V4.2 is PREVIEW VERSION!\n";
		std::cout << "Start HFNAR v4.2 (Preview: Generative)...\n";
		HFNAR40::upscale40_audio(pInputData, outputData.data(), channels, sampleRate, totalPCMFrameCount, model_filename_vnn4, cutoffs[0]);
		}
	else {
			std::cerr << "Error: Version not found.\n";
			drwav_free(pInputData, NULL);
			return 1;
			}

    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = channels;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 32;

    drwav wavOut;
    if (!drwav_init_file_write(&wavOut, out_filename.c_str(), &format, NULL)) {
        std::cerr << "Error: file cannot be written: " << out_filename << "\n";
        drwav_free(pInputData, NULL);
        return 1;
    }

    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wavOut, totalPCMFrameCount, outputData.data());
    std::cout << "Complete, saved frames: " << framesWritten << " -> " << out_filename << "\n";

    // Освобождение ресурсов
    drwav_uninit(&wavOut);
    drwav_free(pInputData, NULL);

    return 0;
}