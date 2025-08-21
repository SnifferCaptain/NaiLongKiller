#include "NLKiller.hpp"
#include <iostream>
#include <algorithm>
#include <execution>
#include <atomic>

NLKiller::NLKiller(bool verbose, int num_threads) 
    : device_type(DeviceType::CPU), gpu_id(0), model_loaded(false), supports_f16(false), 
      verbose(verbose), is_input_f16(false), num_threads(num_threads) {
}

NLKiller::~NLKiller() {
    // OpenVINO资源会自动释放
}

void NLKiller::setNumThreads(int num_threads) {
    this->num_threads = num_threads;
}

void NLKiller::setDevice(DeviceType device, int gpu_id) {
    this->device_type = device;
    this->gpu_id = gpu_id;
}

std::string NLKiller::getDeviceString() const {
    switch (device_type) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::GPU:
            return (gpu_id == 0) ? "GPU" : "GPU." + std::to_string(gpu_id);
        default:
            return "CPU";
    }
}

void NLKiller::setShape(const std::vector<unsigned int>& shape) {
    this->dynamic_shape = shape;
}

std::string NLKiller::formatShape(const ov::Shape& shape) const {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

std::string NLKiller::getModeString(InferenceMode mode) const {
    switch (mode) {
        case InferenceMode::SYNC_SINGLE:
            return "SYNC_SINGLE";
        case InferenceMode::ASYNC_MULTI:
            return "ASYNC_MULTI";
        case InferenceMode::BATCH_SYNC:
            return "BATCH_SYNC";
        default:
            return "UNKNOWN";
    }
}

bool NLKiller::loadModel(const std::string& model_path) {
    try {
        this->model_path = model_path;
        
        // 读取ONNX模型
        model = core.read_model(model_path);
          // 获取输入输出端口信息
        input_port = model->input();
        output_port = model->output();
        
        // 获取原始的PartialShape来检查动态维度
        ov::PartialShape partial_shape = input_port.get_partial_shape();
        output_shape = output_port.get_shape();
        
        // 检查动态轴并进行安全检查
        bool has_dynamic_shape = false;
        for (size_t i = 0; i < partial_shape.size(); ++i) {
            if (partial_shape[i].is_dynamic()) {
                has_dynamic_shape = true;
                break;
            }
        }
        
        if (has_dynamic_shape && dynamic_shape.empty()) {
            if (verbose) {
                std::cout << "\033[31mError: Model has dynamic shape but no shape was set. Use setShape() first.\033[0m" << std::endl;
            }
            return false;
        }
          // 如果有动态轴设置，则重新设置输入形状
        if (!dynamic_shape.empty()) {
            ov::PartialShape new_shape;
            for (unsigned int dim : dynamic_shape) {
                new_shape.push_back(dim);
            }
            model->reshape({{input_port, new_shape}});
            input_shape = model->input().get_shape();
        } else {
            // 如果没有动态形状，直接获取静态形状
            input_shape = input_port.get_shape();
        }
        
        if (verbose) {
            std::cout << "Model input shape: " << formatShape(input_shape) << std::endl;
            std::cout << "Model output shape: " << formatShape(output_shape) << std::endl;
        }
        
        // 检查设备是否支持FP16
        std::string device_str = getDeviceString();
        
        if (device_type == DeviceType::GPU) {
            try {
                auto supported_properties = core.get_property(device_str, ov::supported_properties);
                for (const auto& property : supported_properties) {
                    if (property == ov::hint::inference_precision.name()) {
                        auto inference_precision = core.get_property(device_str, ov::hint::inference_precision);
                        if (inference_precision == ov::element::f16) {
                            supports_f16 = true;
                        }
                    }
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    std::cout << "Warning: Could not check FP16 support: " << e.what() << std::endl;
                }
            }
        }
        
        // 编译模型
        ov::AnyMap config;
        if (supports_f16 && device_type == DeviceType::GPU) {
            config[ov::hint::inference_precision.name()] = ov::element::f16;
            if (verbose) {
                std::cout << "Using FP16 precision for inference" << std::endl;
            }
        }
        
        compiled_model = core.compile_model(model, device_str, config);
        infer_request = compiled_model.create_infer_request();
        
        // 检查编译后模型的输入精度
        auto compiled_input = compiled_model.input();
        is_input_f16 = (compiled_input.get_element_type() == ov::element::f16);
        
        model_loaded = true;
        if (verbose) {
            std::cout << "Model loaded successfully on device: " << device_str << std::endl;
            std::cout << "Input precision: " << (is_input_f16 ? "FP16" : "FP32") << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cout << "\033[31mError loading model: " << e.what() << "\033[0m" << std::endl;
        model_loaded = false;
        return false;
    }
}

bool NLKiller::supportsF16() const {
    return supports_f16;
}

void NLKiller::preprocessImageF32(YTensor<u_char, 3>& input_image, float* output_data, int batch_idx) {
    int height = input_image.shape(0);
    int width = input_image.shape(1);
    int channels = input_image.shape(2);
    
    // 确保输入图像是3通道
    if (channels != 3) {
        std::cout << "\033[31mError: Input image must have 3 channels (HWC format)\033[0m" << std::endl;
        return;
    }
    
    // 计算批次偏移
    int batch_offset = batch_idx * channels * height * width;    // HWC -> CHW 转换并归一化到 [0, 1]
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int chw_idx = batch_offset + c * height * width + h * width + w;
                output_data[chw_idx] = static_cast<float>(input_image.at(h, w, c)) / 255.0f;
            }
        }
    }
}

void NLKiller::preprocessImageF16(YTensor<u_char, 3>& input_image, ov::float16* output_data, int batch_idx) {
    int height = input_image.shape(0);
    int width = input_image.shape(1);
    int channels = input_image.shape(2);
    
    // 确保输入图像是3通道
    if (channels != 3) {
        std::cout << "\033[31mError: Input image must have 3 channels (HWC format)\033[0m" << std::endl;
        return;
    }
    
    // 计算批次偏移
    int batch_offset = batch_idx * channels * height * width;
    
    // HWC -> CHW 转换并归一化到 [0, 1]
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int chw_idx = batch_offset + c * height * width + h * width + w;
                output_data[chw_idx] = static_cast<ov::float16>(static_cast<float>(input_image.at(h, w, c)) / 255.0f);
            }
        }
    }
}

float NLKiller::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> NLKiller::inferSync(std::vector<YTensor<u_char, 3>>& images) {
    last_run_info.num_threads = 1;
    last_run_info.inference_mode = getModeString(InferenceMode::SYNC_SINGLE);
    last_run_info.device = getDeviceString();
    
    std::vector<float> results;
    results.reserve(images.size());
    double total_inference_time = 0.0;
    
    for (auto& image : images) {
        // 准备输入张量
        int height = image.shape(0);
        int width = image.shape(1);
        int channels = image.shape(2);
        
        ov::Shape input_shape = {1, static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)};
        
        ov::Tensor input_tensor;
        if (is_input_f16) {
            input_tensor = ov::Tensor(ov::element::f16, input_shape);
            ov::float16* input_data = input_tensor.data<ov::float16>();
            preprocessImageF16(image, input_data, 0);
        } else {
            input_tensor = ov::Tensor(ov::element::f32, input_shape);
            float* input_data = input_tensor.data<float>();
            preprocessImageF32(image, input_data, 0);
        }
        
        // 设置输入
        infer_request.set_input_tensor(input_tensor);
        
        // 执行推理并计时
        auto infer_start = std::chrono::high_resolution_clock::now();
        infer_request.infer();
        auto infer_end = std::chrono::high_resolution_clock::now();
        
        total_inference_time += std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        
        // 获取输出
        auto output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<const float>();
        
        // 应用sigmoid并保存结果
        float prob = sigmoid(output_data[0]);
        results.push_back(prob);
    }
    
    last_run_info.results = results;
    last_run_info.inference_time = total_inference_time / 1000.0; // 毫秒转秒
    last_run_info.avg_inference_time_per_image = last_run_info.inference_time / images.size();
    
    return results;
}



std::vector<float> NLKiller::inferAsync(std::vector<YTensor<u_char, 3>>& images) {
    last_run_info.inference_mode = getModeString(InferenceMode::ASYNC_MULTI);
    last_run_info.device = getDeviceString();
    
    std::vector<float> results(images.size());
    
    // 设置线程数
    int actual_threads = num_threads;
    if (actual_threads <= 0) {
        actual_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    actual_threads = std::min(actual_threads, static_cast<int>(images.size()));
    last_run_info.num_threads = actual_threads;
    
    // 创建多个推理请求以支持并发
    std::vector<ov::InferRequest> async_requests;
    for (int i = 0; i < actual_threads; ++i) {
        async_requests.push_back(compiled_model.create_infer_request());
    }
    
    // 使用线程池方式处理图片
    std::atomic<size_t> image_index(0);
    std::vector<std::future<void>> futures;
    
    for (int t = 0; t < actual_threads; ++t) {
        auto future = std::async(std::launch::async, [this, &images, &async_requests, &results, &image_index, t]() {
            auto& request = async_requests[t];
            size_t current_index;
            
            while ((current_index = image_index.fetch_add(1)) < images.size()) {
                auto& image = images[current_index];
                
                // 准备输入张量
                int height = image.shape(0);
                int width = image.shape(1);
                int channels = image.shape(2);
                
                ov::Shape input_shape = {1, static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)};
                
                ov::Tensor input_tensor;
                if (is_input_f16) {
                    input_tensor = ov::Tensor(ov::element::f16, input_shape);
                    ov::float16* input_data = input_tensor.data<ov::float16>();
                    preprocessImageF16(image, input_data, 0);
                } else {
                    input_tensor = ov::Tensor(ov::element::f32, input_shape);
                    float* input_data = input_tensor.data<float>();
                    preprocessImageF32(image, input_data, 0);
                }
                
                // 设置输入并推理（每个线程专用自己的request，无需加锁）
                request.set_input_tensor(input_tensor);
                
                auto infer_start = std::chrono::high_resolution_clock::now();
                request.infer();
                auto infer_end = std::chrono::high_resolution_clock::now();
                
                // 获取输出
                auto output_tensor = request.get_output_tensor();
                const float* output_data = output_tensor.data<const float>();
                
                // 应用sigmoid并保存结果
                results[current_index] = sigmoid(output_data[0]);
            }
        });
        
        futures.push_back(std::move(future));
    }
    
    // 等待所有线程完成
    for (auto& future : futures) {
        future.get();
    }
    
    last_run_info.results = results;
    // 由于并行执行，无法精确计算总推理时间，这里设置为0
    last_run_info.inference_time = 0.0;
    last_run_info.avg_inference_time_per_image = 0.0;
    
    return results;
}



std::vector<float> NLKiller::inferBatch(std::vector<YTensor<u_char, 3>>& images) {
    if (images.empty()) {
        last_run_info.num_threads = 1;
        last_run_info.inference_mode = getModeString(InferenceMode::BATCH_SYNC);
        last_run_info.device = getDeviceString();
        last_run_info.inference_time = 0.0;
        last_run_info.avg_inference_time_per_image = 0.0;
        last_run_info.results.clear();
        return {};
    }
    
    size_t batch_size = images.size();
    const auto& first_image = images[0];
    int height = first_image.shape(0);
    int width = first_image.shape(1);
    int channels = first_image.shape(2);
    
    // 检查所有图像尺寸是否一致
    for (size_t i = 1; i < images.size(); ++i) {
        if (images[i].shape(0) != height || images[i].shape(1) != width || images[i].shape(2) != channels) {
            std::cout << "\033[31mWarning: Images have different dimensions in batch mode, falling back to async inference\033[0m" << std::endl;
            return inferAsync(images);  // 使用异步推理
        }
    }
    
    last_run_info.num_threads = 1;
    last_run_info.inference_mode = getModeString(InferenceMode::BATCH_SYNC);
    last_run_info.device = getDeviceString();
    
    // 准备批量输入张量
    ov::Shape batch_input_shape = {batch_size, static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)};
    
    ov::Tensor input_tensor;
    if (is_input_f16) {
        input_tensor = ov::Tensor(ov::element::f16, batch_input_shape);
        ov::float16* input_data = input_tensor.data<ov::float16>();
        // 预处理所有图像
        for (size_t i = 0; i < images.size(); ++i) {
            preprocessImageF16(images[i], input_data, static_cast<int>(i));
        }
    } else {
        input_tensor = ov::Tensor(ov::element::f32, batch_input_shape);
        float* input_data = input_tensor.data<float>();
        // 预处理所有图像
        for (size_t i = 0; i < images.size(); ++i) {
            preprocessImageF32(images[i], input_data, static_cast<int>(i));
        }
    }
    
    // 设置输入
    infer_request.set_input_tensor(input_tensor);
    
    // 执行推理并计时
    auto infer_start = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto infer_end = std::chrono::high_resolution_clock::now();
    
    // 获取输出
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_data = output_tensor.data<const float>();
    
    // 处理输出结果
    std::vector<float> results;
    results.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        float prob = sigmoid(output_data[i]);
        results.push_back(prob);
    }
    
    last_run_info.results = results;
    last_run_info.inference_time = std::chrono::duration<double>(infer_end - infer_start).count();
    last_run_info.avg_inference_time_per_image = last_run_info.inference_time / batch_size;
    
    return results;
}



std::vector<float> NLKiller::infer(std::vector<YTensor<u_char, 3>>& images, InferenceMode mode) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    if (!model_loaded) {
        std::cout << "\033[31mError: Model not loaded. Please load model first.\033[0m" << std::endl;
        return {};
    }
    
    if (images.empty()) {
        last_run_info.num_threads = 1;
        last_run_info.inference_mode = getModeString(mode);
        last_run_info.device = getDeviceString();
        last_run_info.total_time = 0.0;
        last_run_info.inference_time = 0.0;
        last_run_info.avg_total_time = 0.0;
        last_run_info.avg_inference_time_per_image = 0.0;
        last_run_info.results.clear();
        return {};
    }
    
    // 预处理所有图像
    std::vector<YTensor<u_char, 3>> processed_images;
    processed_images.reserve(images.size());
    
    for (auto& image : images) {
        processed_images.push_back(preprocessInputImage(image));
    }
    
    std::vector<float> results;
    switch (mode) {
        case InferenceMode::SYNC_SINGLE:
            results = inferSync(processed_images);
            break;
        case InferenceMode::ASYNC_MULTI:
            results = inferAsync(processed_images);
            break;
        case InferenceMode::BATCH_SYNC:
            results = inferBatch(processed_images);
            break;
        default:
            std::cout << "\033[31mError: Unknown inference mode\033[0m" << std::endl;
            return {};
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    
    // 更新总的运行时间（包括预处理时间）
    last_run_info.total_time = std::chrono::duration<double>(end_total - start_total).count();
    last_run_info.avg_total_time = last_run_info.total_time / images.size();
    
    return results;
}



ov::Shape NLKiller::getInputShape() const {
    return input_shape;
}

ov::Shape NLKiller::getOutputShape() const {
    return output_shape;
}

bool NLKiller::isModelLoaded() const {
    return model_loaded;
}

YTensor<unsigned char, 3> NLKiller::image2Tensor(int height, int width, int channels, const u_char *data) {
    YTensor<unsigned char, 3> op(height, width, channels);
    std::copy(data, data + height * width * channels, op.data);
    return op;
}

const NLKiller::RunInfo& NLKiller::getLastRunInfo() const {
    return last_run_info;
}

float NLKiller::infer(YTensor<u_char, 3> &image) {
    std::vector<YTensor<u_char, 3>> images = {image};
    auto results = infer(images, InferenceMode::SYNC_SINGLE);
    return results.empty() ? 0.0f : results[0];
}

YTensor<u_char, 3> NLKiller::interpolateImage(
    YTensor<u_char, 3>& input_image,
    int target_height, int target_width,
    InterpolationType type
){
    int src_height = input_image.shape(0);
    int src_width = input_image.shape(1);
    int channels = input_image.shape(2);

    YTensor<u_char, 3> output_image({target_height, target_width, channels});

    float scale_h = static_cast<float>(src_height) / target_height;
    float scale_w = static_cast<float>(src_width) / target_width;
    
    for (int h = 0; h < target_height; ++h) {
        for (int w = 0; w < target_width; ++w) {
            for (int c = 0; c < channels; ++c) {
                if (type == InterpolationType::NEAREST) {
                    // 最近邻插值
                    int src_h = static_cast<int>(h * scale_h + 0.5f);
                    int src_w = static_cast<int>(w * scale_w + 0.5f);
                    src_h = std::min(src_h, src_height - 1);
                    src_w = std::min(src_w, src_width - 1);
                    output_image.at(h, w, c) = input_image.at(src_h, src_w, c);
                } else {
                    // 双线性插值
                    float src_h_f = h * scale_h;
                    float src_w_f = w * scale_w;
                    int src_h1 = static_cast<int>(src_h_f);
                    int src_w1 = static_cast<int>(src_w_f);
                    int src_h2 = std::min(src_h1 + 1, src_height - 1);
                    int src_w2 = std::min(src_w1 + 1, src_width - 1);
                    
                    float dh = src_h_f - src_h1;
                    float dw = src_w_f - src_w1;
                    
                    float val = (1 - dh) * (1 - dw) * input_image.at(src_h1, src_w1, c) +
                               (1 - dh) * dw * input_image.at(src_h1, src_w2, c) +
                               dh * (1 - dw) * input_image.at(src_h2, src_w1, c) +
                               dh * dw * input_image.at(src_h2, src_w2, c);

                    output_image.at(h, w, c) = static_cast<u_char>(val + 0.5f);
                }
            }
        }
    }
    
    return output_image;
}

YTensor<u_char, 3> NLKiller::preprocessInputImage(YTensor<u_char, 3>& input_image) {
    int src_height = input_image.shape(0);
    int src_width = input_image.shape(1);
    int src_channels = input_image.shape(2);
    
    // 获取模型输入尺寸
    int model_height = static_cast<int>(input_shape[2]);
    int model_width = static_cast<int>(input_shape[3]);
    int model_channels = static_cast<int>(input_shape[1]);
    
    // 1. 等比例缩放到最大边长能被网络容下
    float scale = std::min(static_cast<float>(model_height) / src_height, 
                          static_cast<float>(model_width) / src_width);
    
    int resize_height = static_cast<int>(src_height * scale);
    int resize_width = static_cast<int>(src_width * scale);
    
    // 插值缩放
    YTensor<u_char, 3> resized_image = interpolateImage(input_image, resize_height, resize_width);

    // 2. 创建填充后的图像
    YTensor<u_char, 3> padded_image({model_height, model_width, src_channels});
    padded_image.fill(0);  // 填充黑色
    
    // 计算居中偏移
    int offset_h = (model_height - resize_height) / 2;
    int offset_w = (model_width - resize_width) / 2;
    
    // 复制缩放后的图像到中心
    for (int h = 0; h < resize_height; ++h) {
        for (int w = 0; w < resize_width; ++w) {
            for (int c = 0; c < src_channels; ++c) {
                padded_image.at(offset_h + h, offset_w + w, c) = resized_image.at(h, w, c);
            }
        }
    }
    
    // 3. 调整通道数
    YTensor<u_char, 3> final_image({model_height, model_width, model_channels});
    
    if (src_channels == model_channels) {
        // 通道数相同，直接复制
        final_image = padded_image.move();
    } else if (src_channels < model_channels) {        // 通道数不足，复制最后一个通道
        for (int h = 0; h < model_height; ++h) {
            for (int w = 0; w < model_width; ++w) {
                for (int c = 0; c < model_channels; ++c) {
                    int src_c = std::min(c, src_channels - 1);
                    final_image.at(h, w, c) = padded_image.at(h, w, src_c);
                }
            }
        }
    } else {        // 通道数过多，舍弃多余通道
        for (int h = 0; h < model_height; ++h) {
            for (int w = 0; w < model_width; ++w) {
                for (int c = 0; c < model_channels; ++c) {
                    final_image.at(h, w, c) = padded_image.at(h, w, c);
                }
            }
        }
    }
    
    return final_image;
}
