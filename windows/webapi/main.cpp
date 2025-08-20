#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <csignal>
#include "webapi.hpp"

// 全局服务器指针，用于信号处理
std::unique_ptr<httplib::Server> g_server;

// 信号处理函数
void signal_handler(int signal) {
    std::cout << "\n接收到信号 " << signal << "，正在关闭服务器..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char *argv[]) {
    // 设置信号处理
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::ifstream configFile("./config.txt");
    std::string model_path = "./models/yvgg_simplified.onnx";
    NLKiller::DeviceType device = NLKiller::DeviceType::CPU;
    int gpuid = 0;
    bool verbose = true;
    int port = 1145; // 默认端口
    
    if (configFile.is_open()) {
        try {
            std::getline(configFile, model_path); // 读取模型路径
            std::string line;
            std::getline(configFile, line); // 读取设备类型
            if (line == "GPU") {
                device = NLKiller::DeviceType::GPU;
            }
            std::getline(configFile, line); // 读取GPU ID
            gpuid = std::stoi(line);
            std::getline(configFile, line); // 读取是否开启详细日志
            if (line == "false") {
                verbose = false;
            }
            std::getline(configFile, line); // 读取端口号
            if (!line.empty()) {
                port = std::stoi(line);
            }
        }
        catch (const std::exception& e) {
            std::cout << "配置文件读取错误: " << e.what() << "，使用默认配置" << std::endl;
            model_path = "./models/yvgg_simplified.onnx";
            device = NLKiller::DeviceType::CPU;
            gpuid = 0;
            verbose = true;
            port = 1145; // 默认端口
        }
        configFile.close();
    }
    else {
        std::cout << "配置文件 config.txt 未找到，使用默认设置。" << std::endl;
    }

    try {
        // 构建服务器
        g_server = build_nlk_server(model_path, device, gpuid, verbose);
    }
    catch (const std::exception& e) {
        if(verbose) std::cout << "服务器配置失败: " << e.what() << std::endl;
        return -1;
    }

    if (verbose) {
        std::cout << "======================================================" << std::endl
                  << " 推理服务已启动，正在监听: http://127.0.0.1:" << port << std::endl
                  << "======================================================" << std::endl;
        std::cout << "API调用格式:" << std::endl;
        std::cout << "原始图像json数据: \n{\n\t\"shape\": [height, width, channels],// 整数" <<
                                          "\n\t\"image_data\": [...]                // 使用base64编码的图片像素数据，从坐上到右下，RGB排布\n}" << std::endl;
        std::cout << "按 Ctrl+C 停止服务器" << std::endl;
    }

    // 启动服务器
    bool success = g_server->listen("127.0.0.1", port);
    
    if (!success) {
        if (verbose) std::cout << "Failed to start server on port " << port << std::endl;
        return -1;
    }

    return 0;
}
