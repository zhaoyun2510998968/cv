"""
下载并准备COCO128数据集用于训练
"""

import os
import urllib.request
import zipfile
import argparse
from pathlib import Path

def download_coco128(save_dir='data'):
    """
    下载COCO128数据集
    
    Args:
        save_dir: 保存目录
    """
    print("COCO128数据集下载脚本")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # COCO128数据集链接
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    zip_file = os.path.join(save_dir, 'coco128.zip')
    
    print(f"\n下载链接: {url}")
    print(f"保存位置: {zip_file}")
    
    try:
        print("\n开始下载... （这可能需要几分钟）")
        urllib.request.urlretrieve(url, zip_file, reporthook=download_progress)
        print("\n下载完成！")
        
        # 解压
        print("\n开始解压...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        print("解压完成！")
        
        # 删除zip文件
        os.remove(zip_file)
        print("临时文件已清理")
        
        print("\n数据集已准备完成！")
        print(f"数据集位置: {os.path.join(save_dir, 'coco128')}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n如果下载失败，可以手动下载：")
        print(f"1. 访问: {url}")
        print(f"2. 将压缩包解压到: {save_dir}")

def download_progress(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 // total_size, 100)
    print(f'\r下载进度: {percent}% [{downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]', end='')

def verify_dataset(data_dir='data/coco128'):
    """验证数据集完整性"""
    print("\n验证数据集...")
    
    required_dirs = ['images/train2017', 'images/val2017', 'labels/train2017', 'labels/val2017']
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if os.path.exists(full_path):
            file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
            print(f"✓ {dir_path}: {file_count} 个文件")
        else:
            print(f"✗ {dir_path}: 不存在")
            all_ok = False
    
    if all_ok:
        print("\n✓ 数据集验证成功！")
    else:
        print("\n✗ 数据集验证失败，请检查下载是否完成")
    
    return all_ok

def main():
    parser = argparse.ArgumentParser(description='下载COCO128数据集')
    parser.add_argument('--save-dir', type=str, default='data',
                        help='数据集保存目录')
    parser.add_argument('--verify', action='store_true',
                        help='验证现有数据集')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(os.path.join(args.save_dir, 'coco128'))
    else:
        download_coco128(args.save_dir)
        verify_dataset(os.path.join(args.save_dir, 'coco128'))

if __name__ == '__main__':
    main()
