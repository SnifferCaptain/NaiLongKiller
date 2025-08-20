#!/usr/bin/env python3
"""
测试客户端 - 演示如何调用重构后的WebAPI服务
"""

import requests
import json
import base64
import numpy as np
from PIL import Image
import io
import os

class WebAPIClient:
    def __init__(self, base_url="http://127.0.0.1:1145"):
        self.base_url = base_url

    def test_with_raw_data(self, image_path):
        """使用原始图像数据进行测试（整数数组格式）"""
        try:
            # 使用PIL读取图像并转换为RGB
            image = Image.open(image_path).convert('RGB')
            
            # 获取图像尺寸
            width, height = image.size
            
            # 转换为numpy数组 (PIL格式是HWC)
            image_array = np.array(image)
            
            # 获取形状信息
            shape = [height, width, 3]  # HWC格式
            
            # 将图像数据展平为一维数组
            image_data = image_array.flatten().tolist()
            
            # 构造请求
            payload = {
                "shape": shape,
                "image_data": image_data
            }
            
            # 发送请求
            response = requests.post(f"{self.base_url}/infer", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("原始数据测试（整数数组格式）成功:")
                print(f"  图像尺寸: {shape}")
                print(f"  数据大小: {len(image_data)}")
                print(f"  数据类型: 整数数组")
                print(f"  结果: {result.get('result')}")
                print(f"  总时间: {result.get('total_time_ms')}ms")
                print(f"  推理时间: {result.get('inference_time_ms')}ms")
                print(f"  设备: {result.get('device')}")
                return result
            else:
                print(f"原始数据测试失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"原始数据测试异常: {e}")
            return None

    def test_with_base64_data(self, image_path):
        """使用Base64编码数据进行测试"""
        try:
            # 使用PIL读取图像并转换为RGB
            image = Image.open(image_path).convert('RGB')
            
            # 获取图像尺寸
            width, height = image.size
            
            # 转换为numpy数组 (PIL格式是HWC)
            image_array = np.array(image)
            
            # 获取形状信息
            shape = [height, width, 3]  # HWC格式
            
            # 将图像数据转换为字节数组并Base64编码
            pixel_bytes = image_array.flatten().tobytes()
            image_data_b64 = base64.b64encode(pixel_bytes).decode('utf-8')
            
            # 构造请求
            payload = {
                "shape": shape,
                "image_data": image_data_b64
            }
            
            # 发送请求
            response = requests.post(f"{self.base_url}/infer", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("Base64数据测试成功:")
                print(f"  图像尺寸: {shape}")
                print(f"  Base64长度: {len(image_data_b64)}")
                print(f"  数据类型: Base64字符串")
                print(f"  结果: {result.get('result')}")
                print(f"  总时间: {result.get('total_time_ms')}ms")
                print(f"  推理时间: {result.get('inference_time_ms')}ms")
                print(f"  设备: {result.get('device')}")
                return result
            else:
                print(f"Base64数据测试失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"Base64数据测试异常: {e}")
            return None
    
    def test_with_synthetic_data(self, height=224, width=224):
        """使用合成数据进行测试（整数数组格式）"""
        try:
            # 生成合成的RGB图像数据
            # 创建一个渐变图像作为测试
            image_data = []
            for h in range(height):
                for w in range(width):
                    # 创建简单的渐变模式
                    r = int((h / height) * 255)
                    g = int((w / width) * 255)
                    b = int(((h + w) / (height + width)) * 255)
                    image_data.extend([r, g, b])
            
            # 构造请求
            payload = {
                "shape": [height, width, 3],
                "image_data": image_data
            }
            
            # 发送请求
            response = requests.post(f"{self.base_url}/infer", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("合成数据测试（整数数组格式）成功:")
                print(f"  图像尺寸: [{height}, {width}, 3]")
                print(f"  数据大小: {len(image_data)}")
                print(f"  数据类型: 整数数组")
                print(f"  结果: {result.get('result')}")
                print(f"  总时间: {result.get('total_time_ms')}ms")
                print(f"  推理时间: {result.get('inference_time_ms')}ms")
                print(f"  设备: {result.get('device')}")
                return result
            else:
                print(f"合成数据测试失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"合成数据测试异常: {e}")
            return None

    def test_with_synthetic_base64_data(self, height=224, width=224):
        """使用合成Base64数据进行测试"""
        try:
            # 生成合成的RGB图像数据
            image_data = []
            for h in range(height):
                for w in range(width):
                    # 创建简单的渐变模式
                    r = int((h / height) * 255)
                    g = int((w / width) * 255)
                    b = int(((h + w) / (height + width)) * 255)
                    image_data.extend([r, g, b])
            
            # 转换为字节数组并Base64编码
            pixel_bytes = bytes(image_data)
            image_data_b64 = base64.b64encode(pixel_bytes).decode('utf-8')
            
            # 构造请求
            payload = {
                "shape": [height, width, 3],
                "image_data": image_data_b64
            }
            
            # 发送请求
            response = requests.post(f"{self.base_url}/infer", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("合成Base64数据测试成功:")
                print(f"  图像尺寸: [{height}, {width}, 3]")
                print(f"  Base64长度: {len(image_data_b64)}")
                print(f"  数据类型: Base64字符串")
                print(f"  结果: {result.get('result')}")
                print(f"  总时间: {result.get('total_time_ms')}ms")
                print(f"  推理时间: {result.get('inference_time_ms')}ms")
                print(f"  设备: {result.get('device')}")
                return result
            else:
                print(f"合成Base64数据测试失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"合成Base64数据测试异常: {e}")
            return None
    
    def test_invalid_requests(self):
        """测试无效请求的错误处理"""
        print("\n测试错误处理:")
        
        # 测试1: 空请求
        try:
            response = requests.post(f"{self.base_url}/infer", json={})
            print(f"空请求测试: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            print(f"空请求测试异常: {e}")
        
        # 测试2: 无效的shape
        try:
            payload = {
                "shape": [224, 224],  # 只有2维，应该是3维
                "image_data": [255] * (224 * 224 * 3)
            }
            response = requests.post(f"{self.base_url}/infer", json=payload)
            print(f"无效shape测试: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            print(f"无效shape测试异常: {e}")
        
        # 测试3: 数据大小不匹配
        try:
            payload = {
                "shape": [224, 224, 3],
                "image_data": [255] * 1000  # 数据大小不匹配
            }
            response = requests.post(f"{self.base_url}/infer", json=payload)
            print(f"数据大小不匹配测试: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            print(f"数据大小不匹配测试异常: {e}")

        # 测试4: 无效的Base64字符串
        try:
            payload = {
                "shape": [2, 2, 3],
                "image_data": "这不是有效的Base64字符串!"
            }
            response = requests.post(f"{self.base_url}/infer", json=payload)
            print(f"无效Base64测试: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            print(f"无效Base64测试异常: {e}")

        # 测试5: Base64数据大小不匹配
        try:
            # 创建一个小的Base64字符串，但shape说是大图像
            small_data = base64.b64encode(bytes([255, 0, 0] * 4)).decode('utf-8')  # 只有4个像素
            payload = {
                "shape": [100, 100, 3],  # 但声明是100x100
                "image_data": small_data
            }
            response = requests.post(f"{self.base_url}/infer", json=payload)
            print(f"Base64大小不匹配测试: {response.status_code} - {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Base64大小不匹配测试异常: {e}")

def main():
    """主函数"""
    print("=== WebAPI 测试客户端 ===")
    print("测试重构后的HTTP API服务 - 支持两种数据格式")
    
    client = WebAPIClient()
    
    # 测试合成数据 - 整数数组格式
    print("\n1. 合成数据测试（整数数组格式）:")
    client.test_with_synthetic_data(64, 64)  # 使用小图像进行快速测试
    
    # 测试合成数据 - Base64格式
    print("\n2. 合成数据测试（Base64格式）:")
    client.test_with_synthetic_base64_data(64, 64)
    
    # 测试错误处理
    client.test_invalid_requests()
    
    # 如果有真实图像文件，测试两种格式
    print("\n3. 真实图像测试:")
    image_path = "../../assets/guide.png"  # 替换为实际的图像路径
    if os.path.exists(image_path):
        print("\n3a. 真实图像测试（整数数组格式）:")
        client.test_with_raw_data(image_path)
        print("\n3b. 真实图像测试（Base64格式）:")
        client.test_with_base64_data(image_path)
    else:
        print(f"图像文件 {image_path} 不存在，跳过真实图像测试")
    
    print("\n=== 测试完成 ===")
    print("说明:")
    print("- 整数数组格式: image_data 为 [255, 128, 0, ...] 的数组")
    print("- Base64格式: image_data 为 'iVBORw0KGgoAAAA...' 的字符串")

if __name__ == "__main__":
    main()
