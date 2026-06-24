"""
评估脚本 - 在测试集上评估 Faster R-CNN 模型
输出: mAP@50, mAP@75, mAP50:95, Precision, Recall, FPS, 不同目标尺寸 mAP
"""
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import Config
from dataset import FishDataset, get_transform
from model import FasterRCNNModel


def collate_fn(batch):
    images, targets = [], []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


def compute_iou(box1, box2):
    """计算 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def evaluate():
    """在测试集上评估模型"""
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    model = FasterRCNNModel(num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(config.MODEL_DIR / "best_model.pth", map_location=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"模型加载完成: best_model.pth")

    # 测试集
    test_dataset = FishDataset(
        data_dir=config.DATA_DIR,
        split='test',
        transform=get_transform('test')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"测试集: {len(test_dataset)} 张图像")

    # mAP 计算器
    map_metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    # Precision/Recall 统计
    all_tp, all_fp, all_fn = 0, 0, 0

    # 推理速度统计
    inference_times = []

    # 评估
    print("\n开始评估...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images_d = [img.to(device) for img in images]
            targets_d = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 计时
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()

            preds = model.model(images_d)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)

            # 格式化
            preds_fmt = [{'boxes': p['boxes'].cpu(), 'scores': p['scores'].cpu(), 'labels': p['labels'].cpu()} for p in preds]
            targets_fmt = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets_d]

            map_metric.update(preds_fmt, targets_fmt)

            # 统计 TP/FP/FN 用于计算 Precision/Recall
            for pred, target in zip(preds_fmt, targets_fmt):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                gt_boxes = target['boxes']

                # 按置信度排序，只取置信度 > 0.5 的
                if len(pred_scores) > 0:
                    keep = pred_scores > 0.5
                    pred_boxes = pred_boxes[keep]

                matched_gt = set()
                tp = 0
                for pb in pred_boxes:
                    best_iou = 0
                    best_j = -1
                    for j, gb in enumerate(gt_boxes):
                        if j not in matched_gt:
                            iou = compute_iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_j = j
                    if best_iou >= 0.5 and best_j >= 0:
                        tp += 1
                        matched_gt.add(best_j)

                all_tp += tp
                all_fp += len(pred_boxes) - tp
                all_fn += len(gt_boxes) - len(matched_gt)

            if (i + 1) % 100 == 0:
                print(f"  已评估 {i+1}/{len(test_dataset)}")

    # 计算指标
    result = map_metric.compute()

    # FPS
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time

    # 计算 Precision 和 Recall
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # 输出结果
    print("\n" + "=" * 50)
    print("测试集评估结果 (best_model.pth)")
    print("=" * 50)
    print(f"  mAP@50:      {result['map_50'].item():.4f}")
    print(f"  mAP@75:      {result['map_75'].item():.4f}")
    print(f"  mAP@50:95:   {result['map'].item():.4f}")
    print()
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  TP: {all_tp} | FP: {all_fp} | FN: {all_fn}")
    print()

    # 不同目标尺寸
    print("  不同目标尺寸 mAP:")
    print(f"    Small:     {result['map_small'].item():.4f}")
    print(f"    Medium:    {result['map_medium'].item():.4f}")
    print(f"    Large:     {result['map_large'].item():.4f}")
    print()

    # 推理速度
    print(f"  平均推理时间: {avg_time*1000:.1f} ms")
    print(f"  FPS:         {fps:.1f}")
    print("=" * 50)

    # 保存结果
    results_text = f"""
测试集评估结果 (best_model.pth)
================================
mAP@50:      {result['map_50'].item():.4f}
mAP@75:      {result['map_75'].item():.4f}
mAP@50:95:   {result['map'].item():.4f}

Precision:   {precision:.4f}
Recall:      {recall:.4f}
F1:          {f1:.4f}
TP: {all_tp} | FP: {all_fp} | FN: {all_fn}

不同目标尺寸 mAP:
  Small:     {result['map_small'].item():.4f}
  Medium:    {result['map_medium'].item():.4f}
  Large:     {result['map_large'].item():.4f}

推理速度:
  平均推理时间: {avg_time*1000:.1f} ms
  FPS:         {fps:.1f}
================================
"""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "test_results.txt", "w") as f:
        f.write(results_text)
    print(f"\n结果已保存至: {config.OUTPUT_DIR / 'test_results.txt'}")


if __name__ == '__main__':
    evaluate()
