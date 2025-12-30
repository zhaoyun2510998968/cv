"""
模型评估脚本
评估模型在测试集上的性能
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict

class ModelEvaluator:
    def __init__(self, model_path, device='0'):
        """
        初始化评估器
        
        Args:
            model_path: 模型权重文件路径
            device: 使用的设备
        """
        self.model = YOLO(model_path)
        self.device = device
        self.results = defaultdict(list)
    
    def evaluate_on_dataset(self, data_path, conf_threshold=0.5):
        """
        在数据集上评估模型
        
        Args:
            data_path: 数据集yaml配置路径
            conf_threshold: 置信度阈值
            
        Returns:
            metrics: 评估指标
        """
        print("\n" + "="*60)
        print("开始模型评估...")
        print("="*60 + "\n")
        
        metrics = self.model.val(
            data=data_path,
            conf=conf_threshold,
            device=self.device,
            save_json=True
        )
        
        return metrics
    
    def print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        
        if hasattr(metrics, 'box'):
            print(f"\n目标检测指标:")
            print(f"  mAP50:      {metrics.box.map50:.4f}")
            print(f"  mAP50-95:   {metrics.box.map:.4f}")
            print(f"  Precision:  {metrics.box.mp:.4f}")
            print(f"  Recall:     {metrics.box.mr:.4f}")
        
        print("\n" + "="*60 + "\n")
    
    def plot_results(self, results_dir, save_path='results/evaluation.png'):
        """
        绘制训练结果
        
        Args:
            results_dir: 训练结果目录
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 找到结果文件
        results_files = list(Path(results_dir).glob('**/results.csv'))
        
        if not results_files:
            print(f"未找到结果文件在 {results_dir}")
            return
        
        results_csv = results_files[0]
        print(f"\n读取结果文件: {results_csv}")
        
        # 读取数据
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 损失曲线
            if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
                axes[0, 0].plot(df['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(df['val/box_loss'], label='Val Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Box Loss Curves')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 精度曲线
            if 'metrics/mAP50' in df.columns:
                axes[0, 1].plot(df['metrics/mAP50'], label='mAP50', marker='o')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP50')
                axes[0, 1].set_title('mAP50 Curve')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 总体损失
            if 'train/loss' in df.columns and 'val/loss' in df.columns:
                axes[1, 0].plot(df['train/loss'], label='Train Loss')
                axes[1, 0].plot(df['val/loss'], label='Val Loss')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Total Loss Curves')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 学习率
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['lr/pg0'], label='Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
            plt.close()
            
        except ImportError:
            print("需要 pandas 库来绘制结果图表")
        except Exception as e:
            print(f"绘制图表出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    
    parser.add_argument('--model', type=str, default='results/train_results/weights/best.pt',
                        help='模型权重文件路径')
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='数据集配置文件')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='使用的GPU设备')
    parser.add_argument('--results-dir', type=str, default='results/train_results',
                        help='训练结果目录（用于绘制曲线）')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = ModelEvaluator(args.model, args.device)
    
    # 评估模型
    metrics = evaluator.evaluate_on_dataset(args.data, args.conf)
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 绘制结果
    evaluator.plot_results(args.results_dir)

if __name__ == '__main__':
    main()
