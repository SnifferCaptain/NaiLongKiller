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

// OpenVINO������������
class NLKiller {
public:
    // OpenVINO����ģʽö��
    enum class InferenceMode {
        SYNC_SINGLE,      // ͬ����������
        ASYNC_MULTI,      // �첽���߳�����
        BATCH_SYNC        // ����ͬ������
    };

    // �豸����ö��
    enum class DeviceType {
        CPU,              // CPU����
        GPU               // GPU����gpu_idĬ��Ϊ0
    };

    // ������Ϣ�ṹ��
    struct RunInfo {
        std::vector<float> results;              // ������
        double total_time;                       // ���к�ʱ���룩
        double avg_total_time;                   // ����ƽ����ʱ���룩
        double inference_time;                   // �����ʱ���룩
        double avg_inference_time_per_image;     // ƽ��ÿ��ͼ�������ʱ���룩
        int num_threads;                         // �����߳���
        std::string inference_mode;              // ����ģʽ
        std::string device;                      // �����豸
    };
public:    
    // ���캯����verbose������ϸ�����num_threads�����첽�����߳���
    NLKiller(bool verbose = false, int num_threads = 1);

    // ��������
    ~NLKiller();

    // @brief ���������豸
    // @param device �豸����
    // @param gpu_id GPU���
    void setDevice(DeviceType device, int gpu_id = 0);

    // @brief ���ö�̬����״
    // @param shape ��̬�����״����
    void setShape(const std::vector<unsigned int> &shape);

    // @brief ����ONNXģ��
    // @param model_path ONNXģ���ļ�·��
    // @return �Ƿ���سɹ�
    bool loadModel(const std::string &model_path);

    // @brief ����Ƿ�֧��FP16����
    // @return �Ƿ�֧��FP16
    bool supportsF16() const;

    // @brief �����첽�����߳���
    // @param num_threads �߳�����-1��ʾʹ��Ӳ��������
    void setNumThreads(int num_threads);

    // @brief ����ӿڣ�֧������ߴ�����ͼ��
    // @param images ����ͼ������
    // @param mode ����ģʽ
    // @return ÿ��ͼƬ�ĸ���ֵ
    std::vector<float> infer(std::vector<YTensor<u_char, 3>> &images,
                             InferenceMode mode = InferenceMode::SYNC_SINGLE);

    // @brief ����ӿڣ�����ͼ������
    // @param image ����ͼ��
    // @return ͼƬ�ĸ���ֵ
    float infer(YTensor<u_char, 3> &image);

    // @brief ��ȡ�ϴ��������ϸ������Ϣ
    // @return RunInfo�ṹ��ĳ�������
    const RunInfo& getLastRunInfo() const;

    // @brief ��ȡģ��������״
    // @return ģ��������״
    ov::Shape getInputShape() const;

    // @brief ��ȡģ�������״
    // @return ģ�������״
    ov::Shape getOutputShape() const;

    // @brief ���ģ���Ƿ��Ѽ���
    // @return ģ���Ƿ��Ѽ���
    bool isModelLoaded() const;

    // @brief ��ͼ������ת��ΪYTensor��ʽ
    // @param height ͼ��߶�
    // @param width ͼ����
    // @param channels ͼ��ͨ����
    // @param data ͼ������ָ��
    // @return ת�����YTensor����
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
    int num_threads;  // �첽�����߳���
    std::vector<unsigned int> dynamic_shape;  // ��̬����״����
    RunInfo last_run_info;  // �ϴ������������Ϣ
    
    // ģ�����������Ϣ
    ov::Output<ov::Node> input_port;
    ov::Output<ov::Node> output_port;
    ov::Shape input_shape;
    ov::Shape output_shape;

    // ��ֵ����ö��
    enum class InterpolationType {
        NEAREST,    // ����ڲ�ֵ
        BILINEAR    // ˫���Բ�ֵ
    };
    
    // ��ȡ�豸�ַ���
    std::string getDeviceString() const;
    
    // ��ȡ����ģʽ�ַ���
    std::string getModeString(InferenceMode mode) const;
    
    // ��ʽ����״���
    std::string formatShape(const ov::Shape& shape) const;
    
    // ͼ���ֵ
    YTensor<u_char, 3> interpolateImage(YTensor<u_char, 3> &input_image,
                                        int target_height, int target_width,
                                        InterpolationType type = InterpolationType::BILINEAR);

    // ͼ��Ԥ����resize + padding + ͨ������
    YTensor<u_char, 3> preprocessInputImage(YTensor<u_char, 3>& input_image);
    
    // ͼ��Ԥ����HWC -> CHW����һ����[0,1]��֧��FP16
    void preprocessImageF32(YTensor<u_char, 3>& input_image, float* output_data, int batch_idx = 0);
    void preprocessImageF16(YTensor<u_char, 3>& input_image, ov::float16* output_data, int batch_idx = 0);
    
    // ���������sigmoidת��
    float sigmoid(float x) const;
    
    // ͬ����������
    std::vector<float> inferSync(std::vector<YTensor<u_char, 3>>& images);
    
    // �첽���߳�����
    std::vector<float> inferAsync(std::vector<YTensor<u_char, 3>>& images);
    
    // ����ͬ������
    std::vector<float> inferBatch(std::vector<YTensor<u_char, 3>>& images);
};