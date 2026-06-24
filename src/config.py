"""
配置文件 - 水下鱼类检测 Mask R-CNN
"""
import os
from pathlib import Path


class Config:
    """项目配置"""
    # 数据集路径
    DATA_DIR = Path("d:/AAA/sea/deepfish_dataset")
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"

    # 模型配置
    NUM_CLASSES = 2  # 1 类鱼 + 背景
    IMAGE_SIZE = (800, 800)

    # 训练配置
    BATCH_SIZE = 1  # RTX 3060 6GB 推荐
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.005  # SGD 推荐稍大学习率
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.9

    # AMP 混合精度
    USE_AMP = True

    # Early Stopping
    EARLY_STOPPING = True
    PATIENCE = 12  # 连续多少轮无提升则停止

    # 检查点间隔
    CHECKPOINT_EVERY = 5  # 每 N 轮保存一次检查点

    # 数据增强
    AUGMENTATION = True
    HORIZONTAL_FLIP = 0.5
    VERTICAL_FLIP = 0.1
    ROTATION = 15
    BRIGHTNESS = 0.2
    CONTRAST = 0.2

    # 设备
    DEVICE = "cuda"  # 或 "cpu"

    # DataLoader
    NUM_WORKERS = 4  # Windows 推荐 4
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2

    # 输出目录
    OUTPUT_DIR = Path("d:/aaa/sea/outputs")
    MODEL_DIR = OUTPUT_DIR / "models"
    LOG_DIR = OUTPUT_DIR / "logs"
    VIS_DIR = OUTPUT_DIR / "visualizations"

    @classmethod
    def create_dirs(cls):
        """创建输出目录"""
        for d in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.VIS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
