#pragma once
#include <string>

bool run_video_upscaler(
	const std::string& model_path,
	const std::string& input_path,
	const std::string& output_path, int target_w, int target_h
);