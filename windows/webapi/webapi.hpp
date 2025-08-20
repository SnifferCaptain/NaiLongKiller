#pragma once

#include <memory>
#include <string>

// �������NLKiller���YTensor
// ע�⣺��Щͷ�ļ���Ҫ�ڱ���ʱ��������ȷ��·��
#include "../NLKiller.hpp" 
#include "../ytensor.hpp"
#include "./httplib.h"
#include "./json.hpp"

using json = nlohmann::json;

/**
 * @brief ����������ͼ������HTTP����
 * @param model_path ONNXģ���ļ���·��
 * @param device Ҫʹ�õ������豸 (CPU or GPU)
 * @param gpuid GPU�豸ID
 * @param verbose �Ƿ�ΪNLKiller������ϸ��־ģʽ
 * @return ����������ָ��
 */
std::unique_ptr<httplib::Server> build_nlk_server(
    const std::string& model_path,
    NLKiller::DeviceType device = NLKiller::DeviceType::CPU,
    int gpuid = 0,
    bool verbose = true
);