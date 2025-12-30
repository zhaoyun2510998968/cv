# YOLO 目标检测课程实验

## 项目简介

本项目实现了基于YOLOv8的目标检测系统，包含数据准备、模型训练、推理和评估等完整工作流程。

**实验目标：**
- 理解YOLO目标检测算法的原理
- 学会使用深度学习框架进行模型训练和优化
- 掌握计算机视觉中的常见评估指标
- 实现完整的目标检测应用

## 项目结构

```
YOLO_Detection/
├── data/                      # 数据集目录
│   ├── coco128/              # COCO128数据集
│   └── data.yaml             # 数据集配置文件
├── scripts/                   # 脚本目录
│   ├── download_dataset.py   # 下载数据集
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   └── evaluate.py           # 评估脚本
├── results/                   # 结果目录
│   ├── train_results/        # 训练结果
│   └── detections/           # 检测结果
├── main.py                   # 主程序
├── requirements.txt          # 依赖包
└── README.md                 # 说明文档
```

## 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.0+ (推荐使用GPU加速)
- 8GB+ 内存
- 4GB+ GPU显存

### 安装步骤

1. **克隆或下载项目**
```bash
cd YOLO_Detection
```

2. **创建虚拟环境（推荐）**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **安装依赖包**
```bash
pip install -r requirements.txt
```

## 快速开始

### 方式1: 运行完整工作流程

```bash
# 执行所有步骤（下载数据集 → 训练 → 评估 → 推理）
python main.py --step all --epochs 100 --batch 16

# 仅执行某个步骤
python main.py --step download    # 下载数据集
python main.py --step train       # 训练模型
python main.py --step evaluate    # 评估模型
python main.py --step inference   # 运行推理
```

### 方式2: 分步骤执行

#### 步骤1：下载数据集

```bash
python scripts/download_dataset.py --save-dir data

# 或验证现有数据集
python scripts/download_dataset.py --save-dir data --verify
```

数据集将下载到 `data/coco128/` 目录。

#### 步骤2：训练模型

```bash
python scripts/train.py \
  --model yolov8n.pt \
  --data data/coco128/data.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --validate
```

**常用参数：**
- `--model`: 模型大小 (yolov8n/s/m/l/x)
- `--epochs`: 训练轮次
- `--batch`: 批大小
- `--imgsz`: 输入图像尺寸
- `--device`: 使用的GPU设备 (0,1,2 或 cpu)

**推荐配置：**
- 小型模型 (n): 快速训练, 准确度一般
- 中型模型 (m): 平衡性能, 推荐
- 大型模型 (l/x): 更高准确度, 需要更多资源

#### 步骤3：评估模型

```bash
python scripts/evaluate.py \
  --model results/train_results/weights/best.pt \
  --data data/coco128/data.yaml \
  --device 0
```

#### 步骤4：运行推理

##### 对单张图像推理
```bash
python scripts/inference.py \
  --model results/train_results/weights/best.pt \
  --source data/test.jpg \
  --source-type image \
  --save results/detections/result.jpg
```

##### 对整个目录推理
```bash
python scripts/inference.py \
  --model results/train_results/weights/best.pt \
  --source data/coco128/images/val2017 \
  --source-type dir \
  --save results/detections
```

##### 对视频推理
```bash
python scripts/inference.py \
  --model results/train_results/weights/best.pt \
  --source video.mp4 \
  --source-type video \
  --save results/detections/output.mp4
```

##### 实时摄像头推理
```bash
python scripts/inference.py \
  --model results/train_results/weights/best.pt \
  --source-type webcam
```

## 核心概念

### YOLO算法简介

YOLO (You Only Look Once) 是一个实时目标检测算法：
- **单阶段检测**: 直接从整张图像预测目标位置和类别
- **高效快速**: 适合实时应用
- **端到端训练**: 直接优化最终检测结果

### 关键指标

| 指标 | 说明 |
|------|------|
| **mAP** | 平均精度均值，综合评估检测性能 |
| **mAP50** | IoU=0.5时的mAP |
| **mAP50-95** | IoU从0.5到0.95的平均mAP |
| **Precision** | 检测正确的比例（准确率） |
| **Recall** | 检测到的真实目标的比例（召回率） |
| **FPS** | 每秒处理帧数（速度） |

### 数据集配置

`data.yaml` 文件示例：
```yaml
path: ../data/coco128           # 数据集根目录
train: images/train2017         # 训练集
val: images/val2017             # 验证集
nc: 80                          # 类别数
names: ['person', 'car', ...]  # 类别名称
```

## 实验要求

### 基础要求
- ✓ 数据集准备（使用COCO128）
- ✓ 模型训练和优化
- ✓ 模型评估和结果分析
- ✓ 推理演示

### 进阶要求
- [ ] 尝试不同的模型大小 (n/s/m/l/x)
- [ ] 数据增强效果分析
- [ ] 超参数调优
- [ ] 自定义数据集训练
- [ ] 模型压缩和优化

## 常见问题

### Q: 如何使用自定义数据集？

A: 准备符合YOLO格式的数据集，修改 `data.yaml` 文件的路径即可。

### Q: 内存不足怎么办？

A: 
- 减小批大小: `--batch 8`
- 减小图像大小: `--imgsz 416`
- 使用更小的模型: `--model yolov8n.pt`

### Q: 训练速度很慢怎么办？

A:
- 检查GPU是否正确识别
- 使用更小的模型或数据集
- 减少验证频率

### Q: 如何保存训练过程中的中间模型？

A: 训练脚本已自动保存，检查 `results/train_results/weights/` 目录。

## 实验总结

通过本实验，你将学会：

1. **深度学习框架**: PyTorch, Ultralytics等工具的使用
2. **计算机视觉**: 目标检测的基本原理和算法
3. **模型训练**: 超参数调优、数据增强等技巧
4. **性能评估**: 如何正确评估检测模型的性能
5. **实际应用**: 从模型到可用系统的完整流程

## 参考资源

- [YOLO官方网站](https://docs.ultralytics.com/)
- [COCO数据集](https://cocodataset.org/)
- [PyTorch官网](https://pytorch.org/)
- [YOLOv5/v8论文](https://arxiv.org/)

## 许可证

MIT License

## 联系方式

如有问题，请查看官方文档或提交Issue。

---

**祝实验顺利！**

> 最后修改: 2025年12月28日
