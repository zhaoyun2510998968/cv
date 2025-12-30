"""
可视化训练曲线
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练结果
results_csv = 'results/train_results3/results.csv'
df = pd.read_csv(results_csv)

# 去掉列名中的空格
df.columns = df.columns.str.strip()

print("可用的列:", df.columns.tolist())
print(f"\n训练了 {len(df)} 个epochs")

# 创建2x3的子图布局
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('YOLOv8 训练曲线可视化', fontsize=16, fontweight='bold')

# 1. Box Loss曲线
if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
    axes[0, 0].plot(df.index, df['train/box_loss'], label='训练集', linewidth=2, color='#2E86AB')
    axes[0, 0].plot(df.index, df['val/box_loss'], label='验证集', linewidth=2, color='#A23B72')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Box Loss', fontsize=12)
    axes[0, 0].set_title('边界框损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

# 2. Class Loss曲线
if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
    axes[0, 1].plot(df.index, df['train/cls_loss'], label='训练集', linewidth=2, color='#2E86AB')
    axes[0, 1].plot(df.index, df['val/cls_loss'], label='验证集', linewidth=2, color='#A23B72')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Class Loss', fontsize=12)
    axes[0, 1].set_title('分类损失曲线', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# 3. DFL Loss曲线
if 'train/dfl_loss' in df.columns and 'val/dfl_loss' in df.columns:
    axes[0, 2].plot(df.index, df['train/dfl_loss'], label='训练集', linewidth=2, color='#2E86AB')
    axes[0, 2].plot(df.index, df['val/dfl_loss'], label='验证集', linewidth=2, color='#A23B72')
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('DFL Loss', fontsize=12)
    axes[0, 2].set_title('DFL损失曲线', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')

# 4. mAP曲线
if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
    axes[1, 0].plot(df.index, df['metrics/mAP50(B)'], label='mAP@0.5', 
                    linewidth=2.5, color='#F18F01', marker='o', markersize=3)
    axes[1, 0].plot(df.index, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', 
                    linewidth=2.5, color='#C73E1D', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('mAP', fontsize=12)
    axes[1, 0].set_title('平均精度均值 (mAP)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

# 5. Precision & Recall曲线
if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
    axes[1, 1].plot(df.index, df['metrics/precision(B)'], label='精确率 (Precision)', 
                    linewidth=2.5, color='#06A77D', marker='o', markersize=3)
    axes[1, 1].plot(df.index, df['metrics/recall(B)'], label='召回率 (Recall)', 
                    linewidth=2.5, color='#D62246', marker='s', markersize=3)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('数值', fontsize=12)
    axes[1, 1].set_title('精确率与召回率', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

# 6. 学习率曲线
if 'lr/pg0' in df.columns:
    axes[1, 2].plot(df.index, df['lr/pg0'], linewidth=2.5, color='#6A4C93')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('学习率', fontsize=12)
    axes[1, 2].set_title('学习率变化', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# 保存图表
output_path = 'results/train_results3/training_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ 训练曲线已保存到: {output_path}")

# 显示图表
plt.show()

# 打印最终指标
print("\n" + "="*60)
print("最终训练指标:")
print("="*60)
last_row = df.iloc[-1]
if 'metrics/mAP50(B)' in df.columns:
    print(f"mAP@0.5:        {last_row['metrics/mAP50(B)']:.4f}")
if 'metrics/mAP50-95(B)' in df.columns:
    print(f"mAP@0.5:0.95:   {last_row['metrics/mAP50-95(B)']:.4f}")
if 'metrics/precision(B)' in df.columns:
    print(f"精确率:         {last_row['metrics/precision(B)']:.4f}")
if 'metrics/recall(B)' in df.columns:
    print(f"召回率:         {last_row['metrics/recall(B)']:.4f}")
print("="*60)
