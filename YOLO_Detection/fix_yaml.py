"""创建正确的data.yaml文件"""
import os
import yaml

data_config = {
    'path': 'data/coco128',
    'train': 'images/train2017',
    'val': 'images/val2017',
    'nc': 80,
    'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
              'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
              'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 'oven', 
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
              'hair drier', 'toothbrush']
}

yaml_path = 'data/data.yaml'

with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

print(f"已创建正确的 {yaml_path}")
print(f"类别数: {len(data_config['names'])}")
