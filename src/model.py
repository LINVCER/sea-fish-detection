"""
Faster R-CNN 模型定义（适用于无分割标注的检测任务）
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes=2, pretrained=True):
    """
    获取 Faster R-CNN 模型

    Args:
        num_classes: 类别数（包括背景）
        pretrained: 是否使用预训练权重

    Returns:
        model: Faster R-CNN 模型
    """
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        min_size=800,
        max_size=1333
    )

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class FasterRCNNModel(nn.Module):
    """封装的 Faster R-CNN 模型"""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = get_model(num_classes, pretrained)

    def forward(self, images, targets=None):
        """
        前向传播

        Args:
            images: 输入图像列表
            targets: 训练时的目标字典列表

        Returns:
            loss_dict: 训练时的损失字典
            detections: 推理时的检测结果列表
        """
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(images)

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            return self.model([image])[0]
