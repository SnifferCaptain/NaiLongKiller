# NaiLongKiller

急速奶龙查杀王

在4060m使用pytorch进行批量推理，速度可达**114514+ 图片/秒！**</br></br>
*推理设置：pytorch2.5.1，yvgg融合模型，batchsize=6144，开启amp加速，准确率: 98.95% 正确率: 93.89% 召回率: 93.51%*

## 编译环境依赖

- **操作系统**: Linux (Ubuntu 18.04+)
- **编译器**: GCC 7.0+ 或 Clang 6.0+ (支持 C++17)
- **依赖库**:
  - Intel OpenVINO 2023.2.0+
  - Qt5 (Core, Widgets)
  - Intel TBB
  - CMake 3.10+

### 安装依赖

```bash
# 安装基本依赖
sudo apt update
sudo apt install build-essential cmake pkg-config

# 安装 Qt5
sudo apt install qtbase5-dev qt5-qmake

# 安装 Intel OpenVINO
# 请按照官方文档安装 OpenVINO: https://docs.openvino.ai/

# 安装 Intel TBB
sudo apt install libtbb-dev
```

### 编译
位于项目根目录下执行以下命令
```bash
mkdir build && cd build
cmake ..
make -j8
```

## 如何操作

### GUI 使用
编译完成后，进入项目根目录下执行以下命令启动程序
```bash
./build/NLKiller
```

1. 启动程序后会自动加载默认模型（helicopter）
2. 点击"打开文件夹"选择包含图像的目录
3. 程序会自动扫描并显示所有支持的图像文件（jpg、png、bmp、tiff、tga）
4. 程序会自动对第一张图片进行推理
5. 调节置信度阈值滑块（默认 0.5）
6. 点击"一键查杀"按钮对所有图像进行批量检测
7. 查看结果表格：✅ 表示正样本，❌ 表示负样本
8. 点击表格中的图像行可以切换预览
9. 点击"刷新"按钮重新应用置信度阈值
10. 点击"导出结果"保存检测结果

**快捷键**: 
- `W/A/↑/←`: 上一张图像
- `S/D/↓/→`: 下一张图像

**模型选择**:
- 质量最高 (helicopter): 检测精度最高
- 平衡 (NLK-s): 精度和速度平衡
- 速度最快 (yvgg): 检测速度最快

### 模型文件

将模型文件放在 `models` 目录下：
- `models/helicopter_simplified.onnx`
- `models/NLK-s_simplified.onnx` 
- `models/yvgg_simplified.onnx`

## 如何使用 NLKiller 类

### 基本使用

```cpp
#include "NLKiller.hpp"

// 创建推理器
NLKiller killer(false, 4);  // verbose=false, 使用异步推理是启用4线程

// 设置设备
killer.setDevice(NLKiller::DeviceType::CPU);// 使用CPU进行推理

// 加载模型
killer.loadModel("model.onnx");// 加载模型，当前模型暂时不支持批量推理。
std::cout << "是否成功加载：" << killer.isModelLoaded() << std::endl;

// 推理单张图像
YTensor<unsigned char, 3> image = NLKiller::image2Tensor(height, width, channels, data);
float result = killer.infer(image);

// 或者使用异步推理
// std::vector<YTensor<unsigned char, 3>> images;
// std::vector<float> results = killer.infer(images, NLKiller::InferenceMode::ASYNC_MULTI);

// 查看推理详细信息
RunInfo runInfo = killer.getLastRunInfo();
/* runinfo结构体包含：
    std::vector<float> results;              // 推理结果
    double total_time;                       // 运行耗时（秒）
    double avg_total_time;                   // 运行平均耗时（秒）
    double inference_time;                   // 推理耗时（秒）
    double avg_inference_time_per_image;     // 平均每张图像推理耗时（秒）
    int num_threads;                         // 推理线程数
    std::string inference_mode;              // 推理模式
    std::string device;                      // 推理设备
*/
std::cout << "total_time: " << runInfo.total_time << std::endl;// 查看总耗时（秒）
```

### 主要接口

```cpp
// 构造函数
NLKiller(bool verbose = false, int num_threads = 1);

// 设备管理
void setDevice(DeviceType device, int gpu_id = 0);
void setNumThreads(int num_threads);

// 模型管理
bool loadModel(const std::string &model_path);
bool isModelLoaded() const;

// 推理接口
float infer(YTensor<u_char, 3> &image);
std::vector<float> infer(std::vector<YTensor<u_char, 3>> &images, InferenceMode mode);

// 工具方法
static YTensor<unsigned char, 3> image2Tensor(int height, int width, int channels, const u_char *data);
const RunInfo& getLastRunInfo() const;
```

### 推理模式

```cpp
enum class InferenceMode {
    SYNC_SINGLE,      // 同步单张推理
    ASYNC_MULTI,      // 异步多线程推理  
    BATCH_SYNC        // 批量同步推理
};
```

### 设备类型

```cpp
enum class DeviceType {
    CPU,              // CPU推理
    GPU               // GPU推理（需要驱动支持，非批量推理还是cpu更快）
};
```
## 模型详情
### yvgg
1、参考了repVGG模型，训练时将卷积层的前后直接残差相连，有助于梯度在训练时的传播，加速训练，改善模型最终效果。推理时将模块融合为卷积层，并使用卷积核的权重进行卷积计算，无需额外的分支开销。
2、大量使用了pw-dw-pw结构，相比标准卷积能够节省计算量，而且相比分组卷积对算子而言，dw卷积以及可以被看做矩阵乘法的pw卷积是被高度优化的，无需额外reshape开销。在amd 680m移动gpu上使用openvino作为推理框架，推理延迟可达0.2ms以内。