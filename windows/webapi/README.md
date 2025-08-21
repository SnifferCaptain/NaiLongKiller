# NaiLongKiller WebAPI

一个基于HTTP协议的图像推理API服务，使用OpenVINO进行AI模型推理。

## ✨ 特性

- 🚀 HTTP RESTful API接口
- 💻 支持CPU和GPU推理
- 🌐 多语言客户端支持（Python、PHP等）
- 📋 详细的错误处理和日志记录
- 🔗 CORS支持，便于前端调用
- ⚡ 高性能异步处理

## 📋 目录

- [快速开始](#🚀-快速开始)
- [API文档](#📖-api文档)
- [配置文件](#⚙️-配置文件)
- [客户端调用示例](#💻-客户端调用示例)
- [错误处理](#❌-错误处理)
- [编译和部署](#🔧-编译和部署)
- [开源依赖](#📚-开源依赖)

## 🚀 快速开始

### 1. 启动服务

```bash
# 确保配置文件和模型文件存在
# config.txt - 配置文件
# yvgg_simplified.onnx - ONNX模型文件

# 运行服务
./webapi.exe
```

### 2. 默认配置

- **服务地址**: `http://127.0.0.1:1145`
- **API端点**: `/infer`
- **协议**: HTTP POST
- **数据格式**: JSON

启动成功后会看到如下输出：
```
======================================================
 推理服务已启动，正在监听: http://127.0.0.1:1145
======================================================
```

## 📖 API文档

### 推理接口

**URL**: `POST /infer`

**Content-Type**: `application/json`

#### 请求格式

**支持两种格式的 image_data：**

**格式1 - Base64字符串 (推荐)**:
```json
{
    "shape": [height, width, channels],
    "image_data": "base64_encoded_pixel_data"
}
```

**格式2 - 整数数组**:
```json
{
    "shape": [height, width, channels],
    "image_data": [pixel_value_array]
}
```

**参数说明**:

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `shape` | `array<int>` | ✅ | 图像尺寸，格式为`[高度, 宽度, 通道数]` |
| `image_data` | `string` 或 `array<int>` | ✅ | **支持两种格式**：<br/>📝 Base64编码的像素数据字符串 (推荐)<br/>🔢 0-255范围的像素值整数数组 |

#### 响应格式

**成功响应** (HTTP 200):
```json
{
    "result": 0.95,
    "total_time_ms": 25.6,
    "inference_time_ms": 18.2,
    "inference_mode": "sync",
    "device": "CPU",
    "num_threads": 8
}
```

**错误响应** (HTTP 400/500):
```json
{
    "error": "错误描述信息"
}
```

#### 图像数据格式说明

**API支持两种 image_data 格式**:

### 📝 格式1：Base64字符串 (推荐)

💡 **Base64格式**: `image_data` 字段为 **Base64编码的字符串**

**数据处理流程**:
1. **原始图像** → RGB像素数组（0-255）
2. **像素排列**: 从左上角到右下角，逐行扫描，RGB格式
3. **Base64编码**: 将像素字节数组编码为Base64字符串
4. **发送给API**: 将Base64字符串作为`image_data`的值

**示例**: 
```json
{
    "shape": [2, 2, 3],
    "image_data": "/wAA/wD//wA="
}
```

### 🔢 格式2：整数数组

**数据处理流程**:
1. **原始图像** → RGB像素数组（0-255）
2. **像素排列**: 从左上角到右下角，逐行扫描，RGB格式
3. **发送给API**: 直接将整数数组作为`image_data`的值

**示例**: 对于一个2×2的RGB图像
```json
{
    "shape": [2, 2, 3],
    "image_data": [255,0,0, 0,255,0, 0,0,255, 255,255,255]
}
```
```
原始图像:  [红色像素] [绿色像素]
          [蓝色像素] [白色像素]

像素数组: [255,0,0, 0,255,0, 0,0,255, 255,255,255]
         ↑红色    ↑绿色    ↑蓝色    ↑白色
```

**共同说明**:
- **像素格式**: RGB（红-绿-蓝），每个像素3个字节
- **像素顺序**: 从左上角开始，逐行从左到右扫描
- **数据长度**: 原始像素数组长度为 `height × width × 3`

#### Base64编码详细说明

**什么是Base64编码？**
Base64是一种用64个可打印字符来表示二进制数据的编码方式。它将每3个字节（24位）的数据编码为4个Base64字符。

**编码过程**:
1. **准备像素数据**: 将图像转换为RGB像素数组，每个像素3个字节（R、G、B）
2. **字节数组**: 按HWC格式排列 `[R?,G?,B?, R?,G?,B?, ...]`
3. **Base64编码**: 将字节数组编码为Base64字符串
4. **发送API**: 将Base64字符串作为`image_data`的值

**重要提醒**: 
- ✅ **正确**: `"image_data": "iVBORw0KGgoAAAANSUhEUgAA..."` (Base64字符串)
- ❌ **错误**: `"image_data": [255, 0, 0, 128, ...]` (数字数组)

**Base64字符集**: `A-Z`, `a-z`, `0-9`, `+`, `/`, `=`(填充)

## ⚙️ 配置文件

创建 `config.txt` 文件来配置服务参数：

```txt
./yvgg_simplified.onnx
CPU
0
true
1145
```

**配置说明**:

| 行号 | 参数 | 描述 | 示例值 |
|------|------|------|--------|
| 1 | 模型路径 | ONNX模型文件路径 | `./yvgg_simplified.onnx` |
| 2 | 设备类型 | 推理设备 | `CPU` 或 `GPU` |
| 3 | GPU ID | GPU设备编号（仅GPU模式） | `0` |
| 4 | 详细日志 | 是否开启详细日志 | `true` 或 `false` |
| 5 | 端口号 | HTTP服务端口 | `1145` |

### 配置示例

**CPU推理配置**:
```txt
./models/my_model.onnx
CPU
0
true
8080
```

**GPU推理配置**:
```txt
./models/my_model.onnx
GPU
0
false
8080
```

## 💻 客户端调用示例

### Python客户端

**方法1：使用Base64格式 (推荐)**
```python
import requests
import json
import base64
import numpy as np
from PIL import Image

def call_webapi_base64(image_path):
    # 读取图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # 转换为numpy数组 (HWC格式)
    image_array = np.array(image)
    shape = [height, width, 3]
    
    # 将图像数据展平为字节数组
    pixel_bytes = image_array.flatten().tobytes()
    
    # Base64编码
    image_data_b64 = base64.b64encode(pixel_bytes).decode('utf-8')
    
    # 构造请求
    payload = {
        "shape": shape,
        "image_data": image_data_b64
    }
    
    # 发送请求
    response = requests.post(
        "http://127.0.0.1:1145/infer", 
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"推理结果: {result['result']}")
        print(f"推理时间: {result['inference_time_ms']}ms")
        return result
    else:
        print(f"请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return None

# 使用示例
result = call_webapi_base64("./nailong.png")
```

**方法2：使用整数数组格式**
```python
import requests
import json
import numpy as np
from PIL import Image

def call_webapi_array(image_path):
    # 读取图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # 转换为numpy数组 (HWC格式)
    image_array = np.array(image)
    shape = [height, width, 3]
    
    # 展平为一维数组并转换为列表
    image_data = image_array.flatten().tolist()
    
    # 构造请求
    payload = {
        "shape": shape,
        "image_data": image_data
    }
    
    # 发送请求
    response = requests.post(
        "http://127.0.0.1:1145/infer", 
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"推理结果: {result['result']}")
        print(f"推理时间: {result['inference_time_ms']}ms")
        return result
    else:
        print(f"请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return None

# 使用示例
result = call_webapi_array("./nailong.png")
```

### PHP客户端

```php
<?php
function callWebAPI($imagePath) {
    // 读取图像
    $image = imagecreatefrompng($imagePath);
    $width = imagesx($image);
    $height = imagesy($image);
    
    // 提取像素数据为字节数组
    $pixelBytes = '';
    for ($y = 0; $y < $height; $y++) {
        for ($x = 0; $x < $width; $x++) {
            $rgb = imagecolorat($image, $x, $y);
            $r = ($rgb >> 16) & 0xFF;
            $g = ($rgb >> 8) & 0xFF;
            $b = $rgb & 0xFF;
            
            // 添加RGB字节到字符串
            $pixelBytes .= chr($r) . chr($g) . chr($b);
        }
    }
    
    // Base64编码
    $imageDataB64 = base64_encode($pixelBytes);
    
    // 构造请求数据
    $payload = [
        'shape' => [$height, $width, 3],
        'image_data' => $imageDataB64
    ];
    
    // 发送HTTP请求
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, "http://127.0.0.1:1145/infer");
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json'
    ]);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode == 200) {
        $result = json_decode($response, true);
        echo "推理结果: " . $result['result'] . "\n";
        echo "推理时间: " . $result['inference_time_ms'] . "ms\n";
        return $result;
    } else {
        echo "请求失败: HTTP $httpCode\n";
        echo "错误信息: $response\n";
        return null;
    }
    
    imagedestroy($image);
}

// 使用示例
$result = callWebAPI("./nailong.png");
?>
```

### JavaScript客户端 (Node.js)

```javascript
const fs = require('fs');
const axios = require('axios');
const sharp = require('sharp');

async function callWebAPI(imagePath) {
    try {
        // 读取并处理图像，确保为RGB格式
        const { data, info } = await sharp(imagePath)
            .ensureAlpha(false)  // 确保没有Alpha通道
            .raw()
            .toBuffer({ resolveWithObject: true });
        
        // 确保是RGB格式 (3通道)
        if (info.channels !== 3) {
            throw new Error(`图像必须是RGB格式，当前通道数: ${info.channels}`);
        }
        
        const shape = [info.height, info.width, 3];
        
        // Base64编码像素数据
        const imageDataB64 = data.toString('base64');
        
        // 构造请求
        const payload = {
            shape: shape,
            image_data: imageDataB64
        };
        
        // 发送请求
        const response = await axios.post(
            'http://127.0.0.1:1145/infer',
            payload,
            {
                headers: { 'Content-Type': 'application/json' }
            }
        );
        
        console.log('推理结果:', response.data.result);
        console.log('推理时间:', response.data.inference_time_ms + 'ms');
        return response.data;
        
    } catch (error) {
        console.error('请求失败:', error.message);
        if (error.response) {
            console.error('错误详情:', error.response.data);
        }
        return null;
    }
}

// 使用示例
callWebAPI('./nailong.png');
```

## ❌ 错误处理

### 常见错误代码

| HTTP状态码 | 错误类型 | 描述 |
|-----------|----------|------|
| 400 | Bad Request | 请求格式错误或参数无效 |
| 500 | Internal Server Error | 服务器内部错误 |

### 常见错误情况

1. **JSON格式错误**
   ```json
   {"error": "Invalid JSON: syntax error"}
   ```

2. **缺少必需参数**
   ```json
   {"error": "Bad Request: JSON body must contain both 'shape' and 'image_data'"}
   ```

3. **参数类型错误**
   ```json
   {"error": "Bad Request: 'shape' must be an array"}
   ```
   ```json
   {"error": "Bad Request: 'image_data' must be an array or Base64 string"}
   ```

4. **数据尺寸不匹配**
   ```json
   {"error": "Internal server error: Data size doesn't match shape"}
   ```

5. **形状参数无效**
   ```json
   {"error": "Internal server error: Shape must be 3D [height, width, channels]"}
   ```

### 调试建议

1. **检查图像格式**: 确保图像为RGB格式，通道数为3
2. **验证数据格式**: 
   - Base64格式：确保`image_data`是有效的Base64字符串
   - 数组格式：确保`image_data`是0-255范围的整数数组
3. **检查数据长度**: 
   - Base64格式：解码后的字节长度应等于`height × width × 3`
   - 数组格式：数组长度应等于`height × width × 3`
4. **验证JSON格式**: 使用JSON验证工具检查请求格式
5. **测试Base64**: 可以用在线Base64解码工具验证编码是否正确

## 🔧 编译和部署

### 环境要求

- **编译器**: Visual Studio 2022 或更高版本
- **C++标准**: C++17或更高（推荐C++20）
- **操作系统**: Windows 10/11
- **依赖库**: OpenVINO Runtime

### 编译步骤

1. **克隆项目**
   ```bash
   git clone <your-repository>
   cd NaiLongKiller/windows/webapi
   ```

2. **安装OpenVINO**
   - 下载并安装OpenVINO Runtime
   - 设置环境变量`INTEL_OPENVINO_DIR`

3. **配置项目**
   - 使用Visual Studio 2022打开`webapi.vcxproj`
   - 确保项目配置为Release模式
   - 验证C++标准设置为C++17或C++20

4. **编译项目**
   ```bash
   # 使用Visual Studio IDE编译
   # 或使用MSBuild命令行
   msbuild webapi.vcxproj /p:Configuration=Release /p:Platform=x64
   ```

5. **准备运行环境**
   - 将编译生成的`webapi.exe`复制到运行目录
   - 准备`config.txt`配置文件
   - 准备ONNX模型文件
   - 确保OpenVINO Runtime库可访问

### 项目结构

```
webapi/
├── main.cpp              # 主程序入口
├── webapi.hpp            # API头文件
├── webapi.cpp            # API实现
├── httplib.h             # HTTP服务器库
├── json.hpp              # JSON处理库
├── config.txt            # 配置文件
├── yvgg_simplified.onnx  # ONNX模型文件
├── test_client.py        # Python测试客户端
└── webapi.vcxproj        # Visual Studio项目文件
```

### 部署建议

1. **生产环境配置**
   - 设置适当的端口号（避免使用默认1145）
   - 关闭详细日志（`verbose = false`）
   - 配置防火墙规则
   - 使用反向代理（如Nginx）进行负载均衡

2. **性能优化**
   - 根据硬件选择CPU或GPU推理
   - 调整OpenVINO线程数
   - 监控内存使用情况

3. **安全考虑**
   - 实现身份验证机制
   - 添加请求频率限制
   - 验证输入数据安全性

## 📚 开源依赖

本项目使用了以下开源库：

### 1. httplib
- **版本**: 最新版
- **许可证**: MIT License
- **用途**: HTTP服务器功能
- **项目地址**: https://github.com/yhirose/cpp-httplib

### 2. nlohmann/json
- **版本**: 最新版
- **许可证**: MIT License
- **用途**: JSON数据解析和生成
- **项目地址**: https://github.com/nlohmann/json

### 3. OpenVINO
- **版本**: 2023.x
- **许可证**: Apache License 2.0
- **用途**: AI模型推理引擎
- **项目地址**: https://github.com/openvinotoolkit/openvino

## 🔧 故障排除

### 常见问题

1. **端口被占用**
   ```
   Failed to start server on port 1145
   ```
   **解决方案**: 修改`config.txt`中的端口号

2. **模型加载失败**
   ```
   模型加载失败: ./yvgg_simplified.onnx
   ```
   **解决方案**: 检查模型文件路径和权限

3. **OpenVINO初始化失败**
   **解决方案**: 
   - 确保OpenVINO Runtime正确安装
   - 检查环境变量配置
   - 验证硬件兼容性

### 性能监控

- 使用`verbose = true`查看详细日志
- 监控推理时间和总时间
- 观察内存使用情况
- 检查CPU/GPU利用率

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 更新日志

- 2025.8.21 修复README的编码问题。
- 2025.8.20 首次更新
---

*最后更新: 2025年8月21日*
