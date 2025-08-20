# NaiLongKiller WebAPI

һ������HTTPЭ���ͼ������API����ʹ��OpenVINO����AIģ������

## ? ����

- ? HTTP RESTful API�ӿ�
- ? ֧��CPU��GPU����
- ? �����Կͻ���֧�֣�Python��PHP�ȣ�
- ? ��ϸ�Ĵ��������־��¼
- ? CORS֧�֣�����ǰ�˵���
- ? �������첽����

## ? Ŀ¼

- [���ٿ�ʼ](#-���ٿ�ʼ)
- [API�ĵ�](#-api�ĵ�)
- [�����ļ�](#-�����ļ�)
- [�ͻ��˵���ʾ��](#-�ͻ��˵���ʾ��)
- [������](#-������)
- [����Ͳ���](#-����Ͳ���)
- [��Դ����](#-��Դ����)

## ? ���ٿ�ʼ

### 1. ��������

```bash
# ȷ�������ļ���ģ���ļ�����
# config.txt - �����ļ�
# yvgg_simplified.onnx - ONNXģ���ļ�

# ���з���
./webapi.exe
```

### 2. Ĭ������

- **�����ַ**: `http://127.0.0.1:1145`
- **API�˵�**: `/infer`
- **Э��**: HTTP POST
- **���ݸ�ʽ**: JSON

�����ɹ���ῴ�����������
```
======================================================
 ������������������ڼ���: http://127.0.0.1:1145
======================================================
```

## ? API�ĵ�

### ����ӿ�

**URL**: `POST /infer`

**Content-Type**: `application/json`

#### �����ʽ

**֧�����ָ�ʽ�� image_data��**

**��ʽ1 - Base64�ַ��� (�Ƽ�)**:
```json
{
    "shape": [height, width, channels],
    "image_data": "base64_encoded_pixel_data"
}
```

**��ʽ2 - ��������**:
```json
{
    "shape": [height, width, channels],
    "image_data": [pixel_value_array]
}
```

**����˵��**:

| ���� | ���� | ���� | ���� |
|------|------|------|------|
| `shape` | `array<int>` | ? | ͼ��ߴ磬��ʽΪ`[�߶�, ���, ͨ����]` |
| `image_data` | `string` �� `array<int>` | ? | **֧�����ָ�ʽ**��<br/>? Base64��������������ַ��� (�Ƽ�)<br/>? 0-255��Χ������ֵ�������� |

#### ��Ӧ��ʽ

**�ɹ���Ӧ** (HTTP 200):
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

**������Ӧ** (HTTP 400/500):
```json
{
    "error": "����������Ϣ"
}
```

#### ͼ�����ݸ�ʽ˵��

**API֧������ image_data ��ʽ**:

### ? ��ʽ1��Base64�ַ��� (�Ƽ�)

?? **Base64��ʽ**: `image_data` �ֶ�Ϊ **Base64������ַ���**

**���ݴ�������**:
1. **ԭʼͼ��** �� RGB�������飨0-255��
2. **��������**: �����Ͻǵ����½ǣ�����ɨ�裬RGB��ʽ
3. **Base64����**: �������ֽ��������ΪBase64�ַ���
4. **���͸�API**: ��Base64�ַ�����Ϊ`image_data`��ֵ

**ʾ��**: 
```json
{
    "shape": [2, 2, 3],
    "image_data": "/wAA/wD//wA="
}
```

### ? ��ʽ2����������

**���ݴ�������**:
1. **ԭʼͼ��** �� RGB�������飨0-255��
2. **��������**: �����Ͻǵ����½ǣ�����ɨ�裬RGB��ʽ
3. **���͸�API**: ֱ�ӽ�����������Ϊ`image_data`��ֵ

**ʾ��**: ����һ��2��2��RGBͼ��
```json
{
    "shape": [2, 2, 3],
    "image_data": [255,0,0, 0,255,0, 0,0,255, 255,255,255]
}
```
```
ԭʼͼ��:  [��ɫ����] [��ɫ����]
          [��ɫ����] [��ɫ����]

��������: [255,0,0, 0,255,0, 0,0,255, 255,255,255]
         ����ɫ    ����ɫ    ����ɫ    ����ɫ
```

**��ͬ˵��**:
- **���ظ�ʽ**: RGB����-��-������ÿ������3���ֽ�
- **����˳��**: �����Ͻǿ�ʼ�����д�����ɨ��
- **���ݳ���**: ԭʼ�������鳤��Ϊ `height �� width �� 3`

#### Base64������ϸ˵��

**ʲô��Base64���룿**
Base64��һ����64���ɴ�ӡ�ַ�����ʾ���������ݵı��뷽ʽ������ÿ3���ֽڣ�24λ�������ݱ���Ϊ4��Base64�ַ���

**�������**:
1. **׼����������**: ��ͼ��ת��ΪRGB�������飬ÿ������3���ֽڣ�R��G��B��
2. **�ֽ�����**: ��HWC��ʽ���� `[R?,G?,B?, R?,G?,B?, ...]`
3. **Base64����**: ���ֽ��������ΪBase64�ַ���
4. **����API**: ��Base64�ַ�����Ϊ`image_data`��ֵ

**��Ҫ����**: 
- ? **��ȷ**: `"image_data": "iVBORw0KGgoAAAANSUhEUgAA..."` (Base64�ַ���)
- ? **����**: `"image_data": [255, 0, 0, 128, ...]` (��������)

**Base64�ַ���**: `A-Z`, `a-z`, `0-9`, `+`, `/`, `=`(���)

## ?? �����ļ�

���� `config.txt` �ļ������÷��������

```txt
./yvgg_simplified.onnx
CPU
0
true
1145
```

**����˵��**:

| �к� | ���� | ���� | ʾ��ֵ |
|------|------|------|--------|
| 1 | ģ��·�� | ONNXģ���ļ�·�� | `./yvgg_simplified.onnx` |
| 2 | �豸���� | �����豸 | `CPU` �� `GPU` |
| 3 | GPU ID | GPU�豸��ţ���GPUģʽ�� | `0` |
| 4 | ��ϸ��־ | �Ƿ�����ϸ��־ | `true` �� `false` |
| 5 | �˿ں� | HTTP����˿� | `1145` |

### ����ʾ��

**CPU��������**:
```txt
./models/my_model.onnx
CPU
0
true
8080
```

**GPU��������**:
```txt
./models/my_model.onnx
GPU
0
false
8080
```

## ? �ͻ��˵���ʾ��

### Python�ͻ���

**����1��ʹ��Base64��ʽ (�Ƽ�)**
```python
import requests
import json
import base64
import numpy as np
from PIL import Image

def call_webapi_base64(image_path):
    # ��ȡͼ��ת��ΪRGB
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # ת��Ϊnumpy���� (HWC��ʽ)
    image_array = np.array(image)
    shape = [height, width, 3]
    
    # ��ͼ������չƽΪ�ֽ�����
    pixel_bytes = image_array.flatten().tobytes()
    
    # Base64����
    image_data_b64 = base64.b64encode(pixel_bytes).decode('utf-8')
    
    # ��������
    payload = {
        "shape": shape,
        "image_data": image_data_b64
    }
    
    # ��������
    response = requests.post(
        "http://127.0.0.1:1145/infer", 
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"������: {result['result']}")
        print(f"����ʱ��: {result['inference_time_ms']}ms")
        return result
    else:
        print(f"����ʧ��: {response.status_code}")
        print(f"������Ϣ: {response.text}")
        return None

# ʹ��ʾ��
result = call_webapi_base64("./nailong.png")
```

**����2��ʹ�����������ʽ**
```python
import requests
import json
import numpy as np
from PIL import Image

def call_webapi_array(image_path):
    # ��ȡͼ��ת��ΪRGB
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # ת��Ϊnumpy���� (HWC��ʽ)
    image_array = np.array(image)
    shape = [height, width, 3]
    
    # չƽΪһά���鲢ת��Ϊ�б�
    image_data = image_array.flatten().tolist()
    
    # ��������
    payload = {
        "shape": shape,
        "image_data": image_data
    }
    
    # ��������
    response = requests.post(
        "http://127.0.0.1:1145/infer", 
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"������: {result['result']}")
        print(f"����ʱ��: {result['inference_time_ms']}ms")
        return result
    else:
        print(f"����ʧ��: {response.status_code}")
        print(f"������Ϣ: {response.text}")
        return None

# ʹ��ʾ��
result = call_webapi_array("./nailong.png")
```

### PHP�ͻ���

```php
<?php
function callWebAPI($imagePath) {
    // ��ȡͼ��
    $image = imagecreatefrompng($imagePath);
    $width = imagesx($image);
    $height = imagesy($image);
    
    // ��ȡ��������Ϊ�ֽ�����
    $pixelBytes = '';
    for ($y = 0; $y < $height; $y++) {
        for ($x = 0; $x < $width; $x++) {
            $rgb = imagecolorat($image, $x, $y);
            $r = ($rgb >> 16) & 0xFF;
            $g = ($rgb >> 8) & 0xFF;
            $b = $rgb & 0xFF;
            
            // ���RGB�ֽڵ��ַ���
            $pixelBytes .= chr($r) . chr($g) . chr($b);
        }
    }
    
    // Base64����
    $imageDataB64 = base64_encode($pixelBytes);
    
    // ������������
    $payload = [
        'shape' => [$height, $width, 3],
        'image_data' => $imageDataB64
    ];
    
    // ����HTTP����
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
        echo "������: " . $result['result'] . "\n";
        echo "����ʱ��: " . $result['inference_time_ms'] . "ms\n";
        return $result;
    } else {
        echo "����ʧ��: HTTP $httpCode\n";
        echo "������Ϣ: $response\n";
        return null;
    }
    
    imagedestroy($image);
}

// ʹ��ʾ��
$result = callWebAPI("./nailong.png");
?>
```

### JavaScript�ͻ��� (Node.js)

```javascript
const fs = require('fs');
const axios = require('axios');
const sharp = require('sharp');

async function callWebAPI(imagePath) {
    try {
        // ��ȡ������ͼ��ȷ��ΪRGB��ʽ
        const { data, info } = await sharp(imagePath)
            .ensureAlpha(false)  // ȷ��û��Alphaͨ��
            .raw()
            .toBuffer({ resolveWithObject: true });
        
        // ȷ����RGB��ʽ (3ͨ��)
        if (info.channels !== 3) {
            throw new Error(`ͼ�������RGB��ʽ����ǰͨ����: ${info.channels}`);
        }
        
        const shape = [info.height, info.width, 3];
        
        // Base64������������
        const imageDataB64 = data.toString('base64');
        
        // ��������
        const payload = {
            shape: shape,
            image_data: imageDataB64
        };
        
        // ��������
        const response = await axios.post(
            'http://127.0.0.1:1145/infer',
            payload,
            {
                headers: { 'Content-Type': 'application/json' }
            }
        );
        
        console.log('������:', response.data.result);
        console.log('����ʱ��:', response.data.inference_time_ms + 'ms');
        return response.data;
        
    } catch (error) {
        console.error('����ʧ��:', error.message);
        if (error.response) {
            console.error('��������:', error.response.data);
        }
        return null;
    }
}

// ʹ��ʾ��
callWebAPI('./nailong.png');
```

## ? ������

### �����������

| HTTP״̬�� | �������� | ���� |
|-----------|----------|------|
| 400 | Bad Request | �����ʽ����������Ч |
| 500 | Internal Server Error | �������ڲ����� |

### �����������

1. **JSON��ʽ����**
   ```json
   {"error": "Invalid JSON: syntax error"}
   ```

2. **ȱ�ٱ������**
   ```json
   {"error": "Bad Request: JSON body must contain both 'shape' and 'image_data'"}
   ```

3. **�������ʹ���**
   ```json
   {"error": "Bad Request: 'shape' must be an array"}
   ```
   ```json
   {"error": "Bad Request: 'image_data' must be an array or Base64 string"}
   ```

4. **���ݳߴ粻ƥ��**
   ```json
   {"error": "Internal server error: Data size doesn't match shape"}
   ```

5. **��״������Ч**
   ```json
   {"error": "Internal server error: Shape must be 3D [height, width, channels]"}
   ```

### ���Խ���

1. **���ͼ���ʽ**: ȷ��ͼ��ΪRGB��ʽ��ͨ����Ϊ3
2. **��֤���ݸ�ʽ**: 
   - Base64��ʽ��ȷ��`image_data`����Ч��Base64�ַ���
   - �����ʽ��ȷ��`image_data`��0-255��Χ����������
3. **������ݳ���**: 
   - Base64��ʽ���������ֽڳ���Ӧ����`height �� width �� 3`
   - �����ʽ�����鳤��Ӧ����`height �� width �� 3`
4. **��֤JSON��ʽ**: ʹ��JSON��֤���߼�������ʽ
5. **����Base64**: ����������Base64���빤����֤�����Ƿ���ȷ

## ? ����Ͳ���

### ����Ҫ��

- **������**: Visual Studio 2022 ����߰汾
- **C++��׼**: C++17����ߣ��Ƽ�C++20��
- **����ϵͳ**: Windows 10/11
- **������**: OpenVINO Runtime

### ���벽��

1. **��¡��Ŀ**
   ```bash
   git clone <your-repository>
   cd NaiLongKiller/windows/webapi
   ```

2. **��װOpenVINO**
   - ���ز���װOpenVINO Runtime
   - ���û�������`INTEL_OPENVINO_DIR`

3. **������Ŀ**
   - ʹ��Visual Studio 2022��`webapi.vcxproj`
   - ȷ����Ŀ����ΪReleaseģʽ
   - ��֤C++��׼����ΪC++17��C++20

4. **������Ŀ**
   ```bash
   # ʹ��Visual Studio IDE����
   # ��ʹ��MSBuild������
   msbuild webapi.vcxproj /p:Configuration=Release /p:Platform=x64
   ```

5. **׼�����л���**
   - ���������ɵ�`webapi.exe`���Ƶ�����Ŀ¼
   - ׼��`config.txt`�����ļ�
   - ׼��ONNXģ���ļ�
   - ȷ��OpenVINO Runtime��ɷ���

### ��Ŀ�ṹ

```
webapi/
������ main.cpp              # ���������
������ webapi.hpp            # APIͷ�ļ�
������ webapi.cpp            # APIʵ��
������ httplib.h             # HTTP��������
������ json.hpp              # JSON�����
������ config.txt            # �����ļ�
������ yvgg_simplified.onnx  # ONNXģ���ļ�
������ test_client.py        # Python���Կͻ���
������ webapi.vcxproj        # Visual Studio��Ŀ�ļ�
```

### ������

1. **������������**
   - �����ʵ��Ķ˿ںţ�����ʹ��Ĭ��1145��
   - �ر���ϸ��־��`verbose = false`��
   - ���÷���ǽ����
   - ʹ�÷��������Nginx�����и��ؾ���

2. **�����Ż�**
   - ����Ӳ��ѡ��CPU��GPU����
   - ����OpenVINO�߳���
   - ����ڴ�ʹ�����

3. **��ȫ����**
   - ʵ�������֤����
   - �������Ƶ������
   - ��֤�������ݰ�ȫ��

## ? ��Դ����

����Ŀʹ�������¿�Դ�⣺

### 1. httplib
- **�汾**: ���°�
- **���֤**: MIT License
- **��;**: HTTP����������
- **��Ŀ��ַ**: https://github.com/yhirose/cpp-httplib

### 2. nlohmann/json
- **�汾**: ���°�
- **���֤**: MIT License
- **��;**: JSON���ݽ���������
- **��Ŀ��ַ**: https://github.com/nlohmann/json

### 3. OpenVINO
- **�汾**: 2023.x
- **���֤**: Apache License 2.0
- **��;**: AIģ����������
- **��Ŀ��ַ**: https://github.com/openvinotoolkit/openvino

## ? �����ų�

### ��������

1. **�˿ڱ�ռ��**
   ```
   Failed to start server on port 1145
   ```
   **�������**: �޸�`config.txt`�еĶ˿ں�

2. **ģ�ͼ���ʧ��**
   ```
   ģ�ͼ���ʧ��: ./yvgg_simplified.onnx
   ```
   **�������**: ���ģ���ļ�·����Ȩ��

3. **OpenVINO��ʼ��ʧ��**
   **�������**: 
   - ȷ��OpenVINO Runtime��ȷ��װ
   - ��黷����������
   - ��֤Ӳ��������

### ���ܼ��

- ʹ��`verbose = true`�鿴��ϸ��־
- �������ʱ�����ʱ��
- �۲��ڴ�ʹ�����
- ���CPU/GPU������

## ? ����

��ӭ�ύIssue��Pull Request���Ľ���Ŀ��

## ������־
2025.8.20 �״θ���

---

*������: 2025��8��20��*
