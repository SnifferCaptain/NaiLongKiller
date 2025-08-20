#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <csignal>
#include "webapi.hpp"

// ȫ�ַ�����ָ�룬�����źŴ���
std::unique_ptr<httplib::Server> g_server;

// �źŴ�����
void signal_handler(int signal) {
    std::cout << "\n���յ��ź� " << signal << "�����ڹرշ�����..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char *argv[]) {
    // �����źŴ���
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::ifstream configFile("./config.txt");
    std::string model_path = "./models/yvgg_simplified.onnx";
    NLKiller::DeviceType device = NLKiller::DeviceType::CPU;
    int gpuid = 0;
    bool verbose = true;
    int port = 1145; // Ĭ�϶˿�
    
    if (configFile.is_open()) {
        try {
            std::getline(configFile, model_path); // ��ȡģ��·��
            std::string line;
            std::getline(configFile, line); // ��ȡ�豸����
            if (line == "GPU") {
                device = NLKiller::DeviceType::GPU;
            }
            std::getline(configFile, line); // ��ȡGPU ID
            gpuid = std::stoi(line);
            std::getline(configFile, line); // ��ȡ�Ƿ�����ϸ��־
            if (line == "false") {
                verbose = false;
            }
            std::getline(configFile, line); // ��ȡ�˿ں�
            if (!line.empty()) {
                port = std::stoi(line);
            }
        }
        catch (const std::exception& e) {
            std::cout << "�����ļ���ȡ����: " << e.what() << "��ʹ��Ĭ������" << std::endl;
            model_path = "./models/yvgg_simplified.onnx";
            device = NLKiller::DeviceType::CPU;
            gpuid = 0;
            verbose = true;
            port = 1145; // Ĭ�϶˿�
        }
        configFile.close();
    }
    else {
        std::cout << "�����ļ� config.txt δ�ҵ���ʹ��Ĭ�����á�" << std::endl;
    }

    try {
        // ����������
        g_server = build_nlk_server(model_path, device, gpuid, verbose);
    }
    catch (const std::exception& e) {
        if(verbose) std::cout << "����������ʧ��: " << e.what() << std::endl;
        return -1;
    }

    if (verbose) {
        std::cout << "======================================================" << std::endl
                  << " ������������������ڼ���: http://127.0.0.1:" << port << std::endl
                  << "======================================================" << std::endl;
        std::cout << "API���ø�ʽ:" << std::endl;
        std::cout << "ԭʼͼ��json����: \n{\n\t\"shape\": [height, width, channels],// ����" <<
                                          "\n\t\"image_data\": [...]                // ʹ��base64�����ͼƬ�������ݣ������ϵ����£�RGB�Ų�\n}" << std::endl;
        std::cout << "�� Ctrl+C ֹͣ������" << std::endl;
    }

    // ����������
    bool success = g_server->listen("127.0.0.1", port);
    
    if (!success) {
        if (verbose) std::cout << "Failed to start server on port " << port << std::endl;
        return -1;
    }

    return 0;
}
