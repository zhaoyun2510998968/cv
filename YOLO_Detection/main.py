"""
YOLO目标检测 - 完整工作流程
支持数据集下载、模型训练、推理和评估
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n{'='*70}")
    print(f"执行: {description}")
    print(f"{'='*70}\n")
    print(f"命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n错误: 命令执行失败!")
        return False
    return True

def setup_environment():
    """设置环境"""
    print("\n" + "="*70)
    print("YOLO目标检测完整工作流程")
    print("="*70)
    print("\n正在检查环境...")
    
    # 检查必要的目录
    dirs = ['data', 'results', 'scripts']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ 创建目录: {dir_name}")
        else:
            print(f"✓ 目录存在: {dir_name}")

def download_dataset():
    """下载数据集"""
    print("\n" + "="*70)
    print("步骤1: 下载数据集")
    print("="*70)
    
    cmd = "python scripts/download_dataset.py --save-dir data"
    return run_command(cmd, "下载COCO128数据集")

def train_model(epochs=100, batch=16, imgsz=640):
    """训练模型"""
    print("\n" + "="*70)
    print("步骤2: 训练模型")
    print("="*70)
    
    cmd = f"python scripts/train.py --model yolov8n.pt --data data/coco128/data.yaml " \
          f"--epochs {epochs} --batch {batch} --imgsz {imgsz} --device 0 --validate"
    
    return run_command(cmd, "使用YOLOv8训练模型")

def evaluate_model():
    """评估模型"""
    print("\n" + "="*70)
    print("步骤3: 评估模型")
    print("="*70)
    
    cmd = "python scripts/evaluate.py --model results/train_results/weights/best.pt " \
          "--data data/coco128/data.yaml --device 0"
    
    return run_command(cmd, "评估模型性能")

def run_inference(source='data/coco128/images/val2017', source_type='dir'):
    """运行推理"""
    print("\n" + "="*70)
    print("步骤4: 运行推理")
    print("="*70)
    
    cmd = f"python scripts/inference.py --model results/train_results/weights/best.pt " \
          f"--source {source} --source-type {source_type} --device 0 --save results/detections"
    
    return run_command(cmd, "对图像/视频运行推理")

def main():
    parser = argparse.ArgumentParser(description='YOLO目标检测完整工作流程')
    
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'download', 'train', 'evaluate', 'inference'],
                        help='执行的步骤')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次')
    parser.add_argument('--batch', type=int, default=16,
                        help='批大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像大小')
    parser.add_argument('--source', type=str, default='data/coco128/images/val2017',
                        help='推理源')
    parser.add_argument('--source-type', type=str, default='dir',
                        help='推理源类型')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 执行步骤
    steps = {
        'download': download_dataset,
        'train': lambda: train_model(args.epochs, args.batch, args.imgsz),
        'evaluate': evaluate_model,
        'inference': lambda: run_inference(args.source, args.source_type)
    }
    
    if args.step == 'all':
        # 按顺序执行所有步骤
        for step_name in ['download', 'train', 'evaluate', 'inference']:
            if not steps[step_name]():
                print(f"\n错误: {step_name} 步骤失败！")
                break
    else:
        # 执行指定步骤
        if not steps[args.step]():
            print(f"\n错误: {args.step} 步骤失败！")
            return 1
    
    print("\n" + "="*70)
    print("✓ 所有步骤完成！")
    print("="*70)
    print("\n结果目录:")
    print("  - 模型权重: results/train_results/weights/best.pt")
    print("  - 检测结果: results/detections/")
    print("  - 训练日志: results/train_results/")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
