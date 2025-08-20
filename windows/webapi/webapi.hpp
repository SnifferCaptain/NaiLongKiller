#pragma once

#include <memory>
#include <string>

// 引入你的NLKiller类和YTensor
// 注意：这些头文件需要在编译时存在于正确的路径
#include "../NLKiller.hpp" 
#include "../ytensor.hpp"
#include "./httplib.h"
#include "./json.hpp"

using json = nlohmann::json;

/**
 * @brief 构建并运行图像推理HTTP服务
 * @param model_path ONNX模型文件的路径
 * @param device 要使用的推理设备 (CPU or GPU)
 * @param gpuid GPU设备ID
 * @param verbose 是否为NLKiller开启详细日志模式
 * @return 服务器智能指针
 */
std::unique_ptr<httplib::Server> build_nlk_server(
    const std::string& model_path,
    NLKiller::DeviceType device = NLKiller::DeviceType::CPU,
    int gpuid = 0,
    bool verbose = true
);