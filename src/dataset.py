"""
数据集类 - 用于加载水下鱼类检测数据
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


class FishDataset(Dataset):
    """水下鱼类检测数据集"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: 数据目录
            split: 'train', 'val', 'test'
            transform: 数据增强变换
        """
        self.data_dir = Path(data_dir) / split
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.transform = transform

        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))

        print(f"加载 {split} 数据集: {len(self.image_files)} 张图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载标注
        label_path = self.labels_dir / img_path.with_suffix('.txt').name
        boxes, labels = self._load_labels(label_path, image.shape[:2])

        # 转换为 Tensor
        image = Image.fromarray(image)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        # 转换为模型需要的格式
        image = transforms.ToTensor()(image)
        _, img_h, img_w = image.shape

        # Bounding Box 越界保护与无效框过滤
        if len(boxes) > 0:
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }

        return image, target

    def _load_labels(self, label_path, img_shape):
        """加载 YOLO 格式标注"""
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 转换为 xmin, ymin, xmax, ymax
                        img_h, img_w = img_shape
                        xmin = (x_center - width/2) * img_w
                        ymin = (y_center - height/2) * img_h
                        xmax = (x_center + width/2) * img_w
                        ymax = (y_center + height/2) * img_h

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id + 1)  # 0 是背景

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return boxes, labels


class Compose:
    """组合多个变换"""
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class HorizontalFlip:
    """水平翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            if len(boxes) > 0:
                w = image.width
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes, labels





class ColorJitter:
    """颜色增强 - 模拟水下颜色漂移"""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, boxes, labels):
        image = self.transform(image)
        return image, boxes, labels


def get_transform(split='train'):
    """获取数据变换"""
    if split == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ])
    else:
        return Compose([])


def get_dataloader(data_dir, split='train', batch_size=4, num_workers=4):
    """创建数据加载器"""
    dataset = FishDataset(
        data_dir=data_dir,
        split=split,
        transform=get_transform(split)
    )

    # 自定义 collate 函数
    def collate_fn(batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        return images, targets

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
