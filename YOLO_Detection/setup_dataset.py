"""复制训练集作为验证集"""
import shutil
import os

data_dir = r'c:\Users\25109\Desktop\first-year\cv课设\YOLO_Detection\data\coco128'

# 复制images
src_img = os.path.join(data_dir, 'images', 'train2017')
dst_img = os.path.join(data_dir, 'images', 'val2017')
if not os.path.exists(dst_img):
    shutil.copytree(src_img, dst_img)
    print(f"已复制 {src_img} -> {dst_img}")

# 复制labels
src_lbl = os.path.join(data_dir, 'labels', 'train2017')
dst_lbl = os.path.join(data_dir, 'labels', 'val2017')
if not os.path.exists(dst_lbl):
    shutil.copytree(src_lbl, dst_lbl)
    print(f"已复制 {src_lbl} -> {dst_lbl}")

print("验证集准备完成！")
