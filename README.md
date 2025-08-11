# YOLO NCNN Python

一个基于 NCNN 推理框架的 YOLO 目标检测 Python 库，支持 YOLOv8、YOLO11 和 YOLO12 模型的高效推理。

## 特性

- 🚀 **高性能推理**: 基于 NCNN 框架，支持 CPU 和 GPU 加速
- 🎯 **多模型支持**: 支持 YOLOv8、YOLO11、YOLO12 等主流 YOLO 模型
- 📦 **简单易用**: 提供简洁的 Python API，易于集成
- 🎨 **可视化功能**: 内置结果可视化和绘制功能
- ⚡ **批量处理**: 支持单张图片和批量图片处理
- 🔧 **灵活配置**: 支持自定义置信度阈值、NMS 阈值等参数

## 安装

### 环境要求

- Python >= 3.12
- NCNN >= 1.0.20250503

### 使用 pip 安装

```bash
pip install yolo-ncnn
```

### 从源码安装

```bash
git clone https://github.com/JasonG98/YOLO.NCNN.PY.git
cd YOLO.NCNN.PY
pip install -e .
```

## 快速开始

### 基本使用

```python
from yolo_ncnn.detection import YOLOv8, YOLO11, YOLO12

# 初始化 YOLOv8 检测器
detector = YOLOv8(
    param_path="models/yolov8n_ncnn_model/model.ncnn.param",
    bin_path="models/yolov8n_ncnn_model/model.ncnn.bin",
    meta_path="models/yolov8n_ncnn_model/metadata.yaml",
    score_threshold=0.35,
    nms_threshold=0.45,
    device="cpu"  # 或 "gpu"
)

# 单张图片检测
result = detector.predict("path/to/image.jpg")

# 批量图片检测
results = detector.predict(["image1.jpg", "image2.jpg", "image3.jpg"])

# 可视化结果
result.draw(save_path="result.jpg")

# 获取检测信息
print(f"检测到 {len(result)} 个目标")
for i, (box, score, class_id) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
    print(f"目标 {i+1}: 类别={result.names[class_id]}, 置信度={score:.3f}, 边界框={box}")
```

### 使用不同的 YOLO 模型

```python
# YOLOv8
detector_v8 = YOLOv8(
    param_path="models/yolov8n_ncnn_model/model.ncnn.param",
    bin_path="models/yolov8n_ncnn_model/model.ncnn.bin",
    meta_path="models/yolov8n_ncnn_model/metadata.yaml"
)

# YOLO11
detector_11 = YOLO11(
    param_path="models/yolo11n_ncnn_model/model.ncnn.param",
    bin_path="models/yolo11n_ncnn_model/model.ncnn.bin",
    meta_path="models/yolo11n_ncnn_model/metadata.yaml"
)

# YOLO12
detector_12 = YOLO12(
    param_path="models/yolo12n_ncnn_model/model.ncnn.param",
    bin_path="models/yolo12n_ncnn_model/model.ncnn.bin",
    meta_path="models/yolo12n_ncnn_model/metadata.yaml"
)
```

### GPU 加速

```python
# 启用 GPU 加速（需要支持 Vulkan 的设备）
detector = YOLOv8(
    param_path="models/yolov8n_ncnn_model/model.ncnn.param",
    bin_path="models/yolov8n_ncnn_model/model.ncnn.bin",
    meta_path="models/yolov8n_ncnn_model/metadata.yaml",
    device="gpu"
)
```

## 模型转换

本库使用 NCNN 格式的模型文件。您需要将原始的 YOLO 模型转换为 NCNN 格式：

### 从 ONNX 转换

```bash
# 安装 NCNN 工具
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake ..
make -j4

# 转换模型
./tools/onnx/onnx2ncnn model.onnx model.param model.bin
```

### 模型文件结构

```
models/
├── yolov8n_ncnn_model/
│   ├── model.ncnn.param    # 模型结构文件
│   ├── model.ncnn.bin      # 模型权重文件
│   └── metadata.yaml       # 模型元数据
├── yolo11n_ncnn_model/
│   ├── model.ncnn.param
│   ├── model.ncnn.bin
│   └── metadata.yaml
└── yolo12n_ncnn_model/
    ├── model.ncnn.param
    ├── model.ncnn.bin
    └── metadata.yaml
```

### metadata.yaml 示例

```yaml
stride: 32
task: detection
names:
  0: person
  1: bicycle
  2: car
  # ... 更多类别
imgsz: [640, 640]
batch: 1
```

## API 参考

### DetectionResult

检测结果类，包含以下属性：

- `orig_img`: 原始图像 (np.ndarray)
- `boxes`: 检测边界框 (np.ndarray, shape: [N, 4])
- `scores`: 检测置信度 (np.ndarray, shape: [N])
- `class_ids`: 类别 ID (np.ndarray, shape: [N])
- `names`: 类别名称列表 (List[str])

#### 方法

- `draw(save_path=None)`: 绘制检测结果
- `__len__()`: 返回检测到的目标数量

### YOLO 检测器参数

- `param_path`: NCNN 模型参数文件路径
- `bin_path`: NCNN 模型权重文件路径
- `meta_path`: 模型元数据文件路径
- `score_threshold`: 置信度阈值 (默认: 0.35)
- `nms_threshold`: NMS 阈值 (默认: 0.45)
- `device`: 设备类型，"cpu" 或 "gpu" (默认: "cpu")

## 性能优化

### CPU 优化

- 使用多线程处理批量图片
- 调整输入图片尺寸以平衡精度和速度
- 合理设置置信度阈值减少后处理时间

### GPU 优化

- 确保设备支持 Vulkan
- 使用较大的批处理大小
- 预热模型以获得稳定的推理时间

## 测试

运行测试用例：

```bash
# 安装测试依赖
pip install pytest

# 运行测试
pytest tests/
```

## 项目结构

```
yolo_ncnn/
├── __init__.py
├── base.py                 # 基础 YOLO 类
├── detection/
│   ├── __init__.py
│   ├── result.py          # 检测结果类
│   ├── yolov8.py          # YOLOv8 实现
│   ├── yolo11.py          # YOLO11 实现
│   └── yolo12.py          # YOLO12 实现
└── utils/
    ├── __init__.py
    ├── data_load.py       # 数据加载工具
    └── visualize.py       # 可视化工具
```

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [NCNN](https://github.com/Tencent/ncnn) - 高性能神经网络推理框架
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 模型实现
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 更新日志

### v0.1.0

- 初始版本发布
- 支持 YOLOv8、YOLO11、YOLO12 模型
- 基础检测和可视化功能
- CPU 和 GPU 推理支持
