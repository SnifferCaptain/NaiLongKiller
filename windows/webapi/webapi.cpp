#include <memory>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "webapi.hpp"

using u_char = unsigned char;

// Base64���뺯��
std::vector<unsigned char> base64_decode(const std::string& input) {
    const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> result;
    
    int val = 0, valb = -8;
    for (unsigned char c : input) {
        if (c == '=') break;
        if (chars.find(c) == std::string::npos) continue;
        
        val = static_cast<int>((val << 6) + chars.find(c));
        valb += 6;
        if (valb >= 0) {
            result.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return result;
}

// ��ԭʼͼ�����ݴ���YTensor
YTensor<u_char, 3> rawDataToYTensor(const std::vector<int>& shape, const std::vector<unsigned char>& data) {
    if (shape.size() != 3) {
        throw std::invalid_argument("Shape must be 3D [height, width, channels]");
    }
    
    int height = shape[0];
    int width = shape[1]; 
    int channels = shape[2];
    
    if (data.size() != height * width * channels) {
        throw std::invalid_argument("Data size doesn't match shape");
    }
    
    YTensor<u_char, 3> tensor(height, width, channels);
    
    // �������ݣ������������ݰ�HWC��ʽ����
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                int idx = h * width * channels + w * channels + c;
                tensor.at(h, w, c) = data[idx];
            }
        }
    }
    
    return tensor;
}

std::unique_ptr<httplib::Server> build_nlk_server(const std::string& model_path, NLKiller::DeviceType device, int gpuid, bool verbose) {
    auto killer_ptr = std::make_shared<NLKiller>(verbose);
    killer_ptr->setDevice(device, gpuid);

    if(verbose) std::cout << "���ڼ���ģ�ͣ�" << model_path << std::endl;
    if (!killer_ptr->loadModel(model_path)) {
        throw std::runtime_error("ģ�ͼ���ʧ��: " + model_path);
    }
    if (verbose) std::cout << "ģ�ͼ��سɹ�" << std::endl;

    auto server = std::make_unique<httplib::Server>();

    // ����CORSͷ
    server->set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // ����OPTIONS����
    server->Options("/infer", [](const httplib::Request&, httplib::Response& res) {
        return;
    });

    // ��������API·��
    server->Post("/infer", [killer_ptr, verbose](const httplib::Request& req, httplib::Response& res) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            json request_json = json::parse(req.body);
            YTensor<u_char, 3> ytensor;
            bool decoded_successfully = false;

            if (!request_json.contains("shape") || !request_json.contains("image_data")){
                if (verbose) std::cout << "Json���ݱ��������'shape'�Լ�'image_data'" << std::endl;
                json error_response;
                error_response["error"] = "Bad Request: JSON body must contain both 'shape' and 'image_data'";
                res.set_content(error_response.dump(), "application/json");
                res.status = 400;
                return;
            }

            if (!request_json["shape"].is_array()){
				if (verbose) std::cout << "Json���ݵ�'shape'����������" << std::endl;
                json error_response;
                error_response["error"] = "Bad Request: 'shape' must be an array";
                res.set_content(error_response.dump(), "application/json");
                res.status = 400;
                return;
            }

            // ���image_data�����ͣ�֧�������Base64�ַ���
            if (!request_json["image_data"].is_array() && !request_json["image_data"].is_string()){
				if (verbose) std::cout << "Json���ݵ�'image_data'�����������Base64�ַ���" << std::endl;
                json error_response;
                error_response["error"] = "Bad Request: 'image_data' must be an array or Base64 string";
                res.set_content(error_response.dump(), "application/json");
                res.status = 400;
                return;
            }

            std::vector<int> shape = request_json["shape"];
            std::vector<unsigned char> image_data;

            if(shape.size() != 3){
                if (verbose) {
                    std::cout << "Json���ݵ�'shape'������3ά���飬����Ϊ[";
                    for (int a = 0; a < shape.size(); a++) {
                        std::cout << shape[a] << " ";
                    }
                    std::cout << "]" << std::endl;
                }
                json error_response;
                error_response["error"] = "Bad Request: 'shape' must be a 3D array";
                res.set_content(error_response.dump(), "application/json");
                res.status = 400;
                return;
            }

            // ����image_data�����ͽ��д���
            std::string type = "";
            if (request_json["image_data"].is_array()) {
                // ���������ʽ
                std::vector<unsigned char> json_data = request_json["image_data"];
                std::swap(json_data, image_data);
                type = "Array";
            } else if (request_json["image_data"].is_string()) {
                // ����Base64�ַ�����ʽ
                std::string base64_data = request_json["image_data"];
                image_data = base64_decode(base64_data);
                type = "Base64";
            }

            ytensor = rawDataToYTensor(shape, image_data);
            

            // ִ������
            killer_ptr->infer(ytensor);
            const NLKiller::RunInfo& run_info = killer_ptr->getLastRunInfo();
            
            // ������Ӧ
            json response_json;
            response_json["result"] = run_info.results.empty() ? 0. : run_info.results[0];
            response_json["total_time_ms"] = run_info.total_time * 1000.;
            response_json["inference_time_ms"] = run_info.inference_time * 1000.;
            response_json["inference_mode"] = run_info.inference_mode;
            response_json["device"] = run_info.device;
            response_json["num_threads"] = run_info.num_threads;
            
            res.set_content(response_json.dump(), "application/json");
            if (verbose) {
                std::cout << "���յ�ͼ����״: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]����ʽ��Base64 " <<
                    "\t������:" << (run_info.results.empty() ? 0. : run_info.results[0]) << std::endl;
            }
            
        } catch (const json::parse_error& e) {
			if (verbose) std::cout << "JSON��������: " << e.what() << std::endl;
            json error_response;
            error_response["error"] = "Invalid JSON: " + std::string(e.what());
            res.set_content(error_response.dump(), "application/json");
            res.status = 400;
        } catch (const std::exception& e) {
			if (verbose) std::cout << "��������ʱ��������: " << e.what() << std::endl;
            json error_response;
            error_response["error"] = "Internal server error: " + std::string(e.what());
            res.set_content(error_response.dump(), "application/json");
            res.status = 500;
        }
    });

    if (verbose) std::cout << "HTTP������������ɣ�·�������á�" << std::endl;
    return server;
}