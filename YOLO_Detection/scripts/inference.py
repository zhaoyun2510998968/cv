"""
YOLO 目标检测推理脚本
支持图像、视频和实时摄像头推理
"""

import os
import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt

class YOLOInference:
    def __init__(self, model_path, conf_threshold=0.5, device='0'):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重文件路径
            conf_threshold: 置信度阈值
            device: 使用的设备
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        print(f"模型已加载: {model_path}")
        print(f"设备: {device}")
    
    def predict_image(self, image_path, save_path=None):
        """
        对单张图像进行推理
        
        Args:
            image_path: 输入图像路径
            save_path: 保存结果的路径
            
        Returns:
            results: 推理结果
        """
        print(f"\n处理图像: {image_path}")
        
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
            save=True if save_path else False,
            save_txt=True if save_path else False,
            project=os.path.dirname(save_path) if save_path else None,
            name=os.path.basename(save_path) if save_path else 'inference'
        )
        
        # 输出检测信息
        for result in results:
            print(f"检测到 {len(result.boxes)} 个目标")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                print(f"  - {self.model.names[cls_id]}: 置信度 {conf:.3f}, 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        
        return results
    
    def predict_video(self, video_path, save_path=None):
        """
        对视频进行推理
        
        Args:
            video_path: 输入视频路径
            save_path: 保存结果的路径
        """
        print(f"\n处理视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"处理进度: {frame_count} 帧")
            
            # 推理
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )
            
            # 绘制结果
            annotated_frame = results[0].plot()
            
            if save_path:
                out.write(annotated_frame)
            
            # 显示进度
            cv2.imshow('YOLO Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n视频处理完成！总帧数: {frame_count}")
    
    def predict_images_batch(self, image_dir, save_dir=None):
        """
        批量处理图像目录
        
        Args:
            image_dir: 图像目录路径
            save_dir: 保存结果的目录
        """
        os.makedirs(save_dir, exist_ok=True) if save_dir else None
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(image_dir) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        print(f"\n找到 {len(image_files)} 张图像")
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            save_path = os.path.join(save_dir, f"result_{idx}.jpg") if save_dir else None
            
            print(f"\n[{idx}/{len(image_files)}] 处理: {image_file}")
            self.predict_image(image_path, save_path)
    
    def visualize_results(self, result, save_path=None):
        """
        可视化推理结果
        
        Args:
            result: 推理结果
            save_path: 保存路径
        """
        # 获取绘制后的图像
        annotated_img = result.plot()
        
        # 显示
        plt.figure(figsize=(16, 10))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"结果已保存: {save_path}")
        
        plt.tight_layout()
        return annotated_img

def main():
    parser = argparse.ArgumentParser(description='YOLO目标检测推理')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='results/train_results/weights/best.pt',
                        help='模型权重文件路径')
    
    # 输入参数
    parser.add_argument('--source', type=str, default='data/test.jpg',
                        help='输入源 (图像/视频/目录路径)')
    parser.add_argument('--source-type', type=str, default='image',
                        choices=['image', 'video', 'dir', 'webcam'],
                        help='输入源类型')
    
    # 推理参数
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='使用的GPU设备')
    
    # 输出参数
    parser.add_argument('--save', type=str, default=None,
                        help='保存结果的路径')
    
    args = parser.parse_args()
    
    # 初始化推理器
    detector = YOLOInference(args.model, args.conf, args.device)
    
    # 根据类型进行推理
    if args.source_type == 'image':
        results = detector.predict_image(args.source, args.save)
        if results:
            detector.visualize_results(results[0], args.save)
    
    elif args.source_type == 'video':
        detector.predict_video(args.source, args.save)
    
    elif args.source_type == 'dir':
        detector.predict_images_batch(args.source, args.save)
    
    elif args.source_type == 'webcam':
        print("启用摄像头推理模式（按'q'退出）")
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = detector.model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            
            annotated_frame = results[0].plot()
            cv2.imshow('YOLO Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
