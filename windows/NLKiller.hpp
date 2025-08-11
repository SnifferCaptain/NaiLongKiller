#pragma once
#include <openvino/openvino.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <functional>
#include <cmath>
#include <string>
#include <chrono>
#include "ytensor.hpp"

using u_char = unsigned char;
using uchar = unsigned char;

// OpenVINO神经网络推理器
class NLKiller {
public:
    // OpenVINO推理模式枚举
    enum class InferenceMode {
        SYNC_SINGLE,      // 同步单张推理
        ASYNC_MULTI,      // 异步多线程推理
        BATCH_SYNC        // 批量同步推理
    };

    // 设备类型枚举
    enum class DeviceType {
        CPU,              // CPU推理
        GPU               // GPU推理，gpu_id默认为0
    };

    // 运行信息结构体
    struct RunInfo {
        std::vector<float> results;              // 推理结果
        double total_time;                       // 运行耗时（秒）
        double avg_total_time;                   // 运行平均耗时（秒）
        double inference_time;                   // 推理耗时（秒）
        double avg_inference_time_per_image;     // 平均每张图像推理耗时（秒）
        int num_threads;                         // 推理线程数
        std::string inference_mode;              // 推理模式
        std::string device;                      // 推理设备
    };
public:    
    // 构造函数，verbose控制详细输出，num_threads设置异步推理线程数
    NLKiller(bool verbose = false, int num_threads = 1);

    // 析构函数
    ~NLKiller();

    // @brief 设置推理设备
    // @param device 设备类型
    // @param gpu_id GPU编号
    void setDevice(DeviceType device, int gpu_id = 0);

    // @brief 设置动态轴形状
    // @param shape 动态轴的形状向量
    void setShape(const std::vector<unsigned int> &shape);

    // @brief 加载ONNX模型
    // @param model_path ONNX模型文件路径
    // @return 是否加载成功
    bool loadModel(const std::string &model_path);

    // @brief 检查是否支持FP16精度
    // @return 是否支持FP16
    bool supportsF16() const;

    // @brief 设置异步推理线程数
    // @param num_threads 线程数，-1表示使用硬件并发数
    void setNumThreads(int num_threads);

    // @brief 推理接口，支持任意尺寸输入图像
    // @param images 输入图像向量
    // @param mode 推理模式
    // @return 每张图片的概率值
    std::vector<float> infer(std::vector<YTensor<u_char, 3>> &images,
                             InferenceMode mode = InferenceMode::SYNC_SINGLE);

    // @brief 推理接口，单张图像输入
    // @param image 输入图像
    // @return 图片的概率值
    float infer(YTensor<u_char, 3> &image);

    // @brief 获取上次推理的详细运行信息
    // @return RunInfo结构体的常量引用
    const RunInfo& getLastRunInfo() const;

    // @brief 获取模型输入形状
    // @return 模型输入形状
    ov::Shape getInputShape() const;

    // @brief 获取模型输出形状
    // @return 模型输出形状
    ov::Shape getOutputShape() const;

    // @brief 检查模型是否已加载
    // @return 模型是否已加载
    bool isModelLoaded() const;

    // @brief 将图像数据转换为YTensor格式
    // @param height 图像高度
    // @param width 图像宽度
    // @param channels 图像通道数
    // @param data 图像数据指针
    // @return 转换后的YTensor对象
    static YTensor<unsigned char, 3> image2Tensor(int height, int width, int channels, const u_char *data);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    
    std::string model_path;
    DeviceType device_type;
    int gpu_id;
    bool model_loaded;
    bool supports_f16;
    bool verbose;
    bool is_input_f16;
    int num_threads;  // 异步推理线程数
    std::vector<unsigned int> dynamic_shape;  // 动态轴形状设置
    RunInfo last_run_info;  // 上次推理的运行信息
    
    // 模型输入输出信息
    ov::Output<ov::Node> input_port;
    ov::Output<ov::Node> output_port;
    ov::Shape input_shape;
    ov::Shape output_shape;

    // 插值类型枚举
    enum class InterpolationType {
        NEAREST,    // 最近邻插值
        BILINEAR    // 双线性插值
    };
    
    // 获取设备字符串
    std::string getDeviceString() const;
    
    // 获取推理模式字符串
    std::string getModeString(InferenceMode mode) const;
    
    // 格式化形状输出
    std::string formatShape(const ov::Shape& shape) const;
    
    // 图像插值
    YTensor<u_char, 3> interpolateImage(YTensor<u_char, 3> &input_image,
                                        int target_height, int target_width,
                                        InterpolationType type = InterpolationType::BILINEAR);

    // 图像预处理：resize + padding + 通道调整
    YTensor<u_char, 3> preprocessInputImage(YTensor<u_char, 3>& input_image);
    
    // 图像预处理：HWC -> CHW，归一化到[0,1]，支持FP16
    void preprocessImageF32(YTensor<u_char, 3>& input_image, float* output_data, int batch_idx = 0);
    void preprocessImageF16(YTensor<u_char, 3>& input_image, ov::float16* output_data, int batch_idx = 0);
    
    // 对输出进行sigmoid转换
    float sigmoid(float x) const;
    
    // 同步单张推理
    std::vector<float> inferSync(std::vector<YTensor<u_char, 3>>& images);
    
    // 异步多线程推理
    std::vector<float> inferAsync(std::vector<YTensor<u_char, 3>>& images);
    
    // 批量同步推理
    std::vector<float> inferBatch(std::vector<YTensor<u_char, 3>>& images);
};