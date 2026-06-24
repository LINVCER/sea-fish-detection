"""
训练脚本 - Faster R-CNN 水下鱼类检测
支持: AMP 混合精度、torchmetrics mAP 评估、Early Stopping、断点恢复
"""
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import FishDataset, get_transform
from model import FasterRCNNModel

# torchmetrics mAP 计算
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def collate_fn(batch):
    """自定义 collate 函数"""
    images, targets = [], []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


class Trainer:
    """训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = FasterRCNNModel(
            num_classes=config.NUM_CLASSES,
            pretrained=True
        ).to(self.device)

        # 优化器 (SGD 对检测模型更稳定)
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )

        # 数据加载器
        self.train_loader = self._get_dataloader('train')
        self.val_loader = self._get_dataloader('val')

        # 学习率调度器 (每 epoch 更新)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )

        # AMP 混合精度
        self.scaler = GradScaler() if config.USE_AMP and self.device.type == 'cuda' else None

        # torchmetrics mAP 计算器
        self.map_metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

        # Early Stopping
        self.best_map = 0.0
        self.no_improve = 0

        # 训练历史
        self.history = {
            'train_loss': [],
            'mAP50': [],
            'mAP50_95': [],
            'learning_rate': []
        }

    def _get_dataloader(self, split):
        """获取数据加载器"""
        dataset = FishDataset(
            data_dir=self.config.DATA_DIR,
            split=split,
            transform=get_transform(split)
        )

        kwargs = {
            'batch_size': self.config.BATCH_SIZE,
            'shuffle': (split == 'train'),
            'num_workers': self.config.NUM_WORKERS,
            'collate_fn': collate_fn,
            'pin_memory': True,
        }

        if self.config.PERSISTENT_WORKERS and self.config.NUM_WORKERS > 0:
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = self.config.PREFETCH_FACTOR

        return DataLoader(dataset, **kwargs)

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}", dynamic_ncols=True)

        for images, targets in pbar:
            # 移动到设备
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            # AMP 混合精度训练
            if self.scaler:
                with autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # 正确顺序: scale → unscale → clip → step
                self.scaler.scale(losses).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{losses.item():.4f}"})

        # 每 epoch 更新学习率
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """验证 - 计算 mAP"""
        # 计算 mAP
        # 切换到 eval 模式获取检测结果
        self.model.model.eval()
        self.map_metric.reset()

        for images, targets in self.val_loader:
            images_d = [img.to(self.device) for img in images]
            targets_d = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # 推理获取检测结果
            with torch.no_grad():
                preds = self.model.model(images_d)

            # 格式化为 torchmetrics 需要的格式
            preds_formatted = []
            for pred in preds:
                preds_formatted.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })

            targets_formatted = []
            for target in targets_d:
                targets_formatted.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })

            self.map_metric.update(preds_formatted, targets_formatted)

        # 计算 mAP
        map_result = self.map_metric.compute()
        mAP50 = map_result['map_50'].item()
        mAP50_95 = map_result['map'].item()

        return mAP50, mAP50_95

    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练 Faster R-CNN 水下鱼类检测模型")
        print(f"设备: {self.device}")
        print(f"AMP: {'开启' if self.scaler else '关闭'}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Image Size: {self.config.IMAGE_SIZE}")
        print("=" * 60)

        start_epoch = 0

        # 尝试恢复训练
        checkpoint_path = self.config.MODEL_DIR / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            print(f"\n发现检查点，恢复训练...")
            start_epoch, self.best_map, self.no_improve = self.load_checkpoint(checkpoint_path)
            print(f"从 Epoch {start_epoch + 1} 继续训练")

        try:
            for epoch in range(start_epoch, self.config.NUM_EPOCHS):
                start_time = time.time()

                # 训练
                train_loss = self.train_epoch(epoch)

                # 验证
                mAP50, mAP50_95 = self.validate()

                # 记录
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_loss)
                self.history['mAP50'].append(mAP50)
                self.history['mAP50_95'].append(mAP50_95)
                self.history['learning_rate'].append(current_lr)

                epoch_time = time.time() - start_time

                print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  mAP@50:     {mAP50:.4f}")
                print(f"  mAP@50:95:  {mAP50_95:.4f}")
                print(f"  LR:         {current_lr:.6f}")

                # Early Stopping 检查 (基于 mAP@50)
                if mAP50 > self.best_map:
                    self.best_map = mAP50
                    self.no_improve = 0
                    self.save_model('best_model.pth')
                    print(f"  ✓ 保存最佳模型 (mAP@50: {mAP50:.4f})")
                else:
                    self.no_improve += 1
                    print(f"  ✗ 无提升 ({self.no_improve}/{self.config.PATIENCE})")

                # 定期保存检查点
                if (epoch + 1) % self.config.CHECKPOINT_EVERY == 0:
                    self.save_checkpoint(epoch + 1, self.best_map, self.no_improve)

                # 保存最新模型
                self.save_model('latest_model.pth')
                self.save_checkpoint(epoch + 1, self.best_map, self.no_improve)

                # 保存训练历史
                self.save_history()

                # Early Stopping
                if self.config.EARLY_STOPPING and self.no_improve >= self.config.PATIENCE:
                    print(f"\nEarly Stopping: 连续 {self.config.PATIENCE} 轮无提升，停止训练")
                    break

            print("\n" + "=" * 60)
            print("训练完成!")
            print(f"最佳 mAP@50: {self.best_map:.4f}")
            print("=" * 60)

        except Exception as e:
            print(f"\n训练异常: {e}")
            print("保存崩溃检查点...")
            self.save_checkpoint(epoch + 1 if 'epoch' in dir() else 0, self.best_map, self.no_improve)
            raise

    def save_model(self, name):
        """保存模型"""
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = self.config.MODEL_DIR / name
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
        }, path)

    def save_checkpoint(self, epoch, best_map, no_improve):
        """保存检查点（支持断点恢复）"""
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = self.config.MODEL_DIR / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'best_map': best_map,
            'no_improve': no_improve,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['best_map'], checkpoint['no_improve']

    def save_history(self):
        """保存训练历史"""
        self.config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = self.config.LOG_DIR / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """主函数"""
    config = Config()
    config.create_dirs()

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
