"""
YOLO 目标检测模型训练脚本
支持YOLOv8, YOLOv5等模型
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 导入ultralytics库
from ultralytics import YOLO

def setup_environment():
    """设置运行环境"""
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def train_yolo(args):
    """
    训练YOLO模型
    
    Args:
        args: 命令行参数
    """
    setup_environment()
    
    # 创建必要的目录
    os.makedirs(args.project, exist_ok=True)
    os.makedirs(os.path.join(args.project, 'runs'), exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"YOLO目标检测训练配置")
    print(f"{'='*50}")
    print(f"模型: {args.model}")
    print(f"数据集: {args.data}")
    print(f"图像大小: {args.imgsz}")
    print(f"训练轮次: {args.epochs}")
    print(f"批大小: {args.batch}")
    print(f"设备: {args.device}")
    print(f"输出目录: {args.project}")
    print(f"{'='*50}\n")
    
    # 加载模型
    print("正在加载模型...")
    model = YOLO(args.model)
    
    # 训练模型
    print("开始训练...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name='train_results',
        save=True,
        save_period=1,
        workers=args.workers,
        close_mosaic=args.close_mosaic,
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    print("\n训练完成！")
    print(f"结果已保存到: {os.path.join(args.project, 'train_results')}")
    
    return model, results

def validate_model(model, args):
    """
    验证模型
    
    Args:
        model: 训练后的模型
        args: 命令行参数
    """
    print(f"\n{'='*50}")
    print("开始模型验证...")
    print(f"{'='*50}\n")
    
    metrics = model.val(data=args.data)
    
    print("\n验证完成！")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='YOLO目标检测模型训练')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='使用的模型 (yolov8n/s/m/l/x)')
    
    # 数据集参数
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='数据集配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次')
    parser.add_argument('--batch', type=int, default=16,
                        help='批大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='0',
                        help='使用的GPU设备 (0,1,2 或 cpu)')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作进程数')
    parser.add_argument('--close-mosaic', type=int, default=10,
                        help='在最后N个epochs关闭mosaic增强')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='results',
                        help='项目目录')
    
    # 功能参数
    parser.add_argument('--validate', action='store_true',
                        help='训练后进行验证')
    
    args = parser.parse_args()
    
    # 执行训练
    model, results = train_yolo(args)
    
    # 执行验证
    if args.validate:
        validate_model(model, args)

if __name__ == '__main__':
    main()
