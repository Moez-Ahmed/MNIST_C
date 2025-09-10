#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

void collectImgData(std::vector<std::vector<float>>& images, std::vector<int>& labels);
void splitData(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, std::vector<std::vector<float>>& trainImages, std::vector<int>& trainLabels, std::vector<std::vector<float>>& testImages, std::vector<int>& testLabels);
