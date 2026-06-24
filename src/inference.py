"""
推理脚本 - Faster R-CNN 水下鱼类检测
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from config import Config
from model import FasterRCNNModel


class FishDetector:
    """鱼类检测器"""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.config = Config()

        # 加载模型
        self.model = FasterRCNNModel(
            num_classes=self.config.NUM_CLASSES,
            pretrained=False
        )

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成: {model_path}")

    def preprocess(self, image):
        """预处理图像"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(pil_image)

    def postprocess(self, predictions, orig_shape, confidence_threshold=0.5):
        """后处理检测结果"""
        results = []
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                box = boxes[i]

                results.append({
                    'box': box.astype(int),
                    'score': float(scores[i]),
                    'label': int(labels[i])
                })

        return results

    def detect(self, image_path, confidence_threshold=0.5):
        """检测单张图像"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        orig_shape = image.shape
        tensor = self.preprocess(image).to(self.device)

        with torch.no_grad():
            predictions = self.model.model([tensor])[0]

        results = self.postprocess(predictions, orig_shape, confidence_threshold)
        return results, image

    def visualize(self, image, results, save_path=None):
        """可视化检测结果"""
        colors = {1: (0, 255, 0)}  # 鱼 - 绿色
        vis_image = image.copy()

        for result in results:
            box = result['box']
            score = result['score']
            label = result['label']

            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)

            label_text = f"Fish: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_image, (box[0], box[1] - h - 10), (box[0] + w, box[1]), color, -1)
            cv2.putText(vis_image, label_text, (box[0], box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        num_fish = len(results)
        avg_score = np.mean([r['score'] for r in results]) if results else 0
        info_text = f"Fish: {num_fish}, Avg Score: {avg_score:.2f}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(str(save_path), vis_image)
            print(f"结果保存至: {save_path}")

        return vis_image


def main():
    """主函数"""
    config = Config()

    model_path = config.MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型")
        return

    detector = FishDetector(model_path, device=config.DEVICE)

    test_dir = config.DATA_DIR / "test" / "images"
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))[:5]

        for img_path in test_images:
            print(f"\n检测: {img_path.name}")
            results, image = detector.detect(img_path, confidence_threshold=0.9)

            save_path = config.VIS_DIR / f"result_{img_path.name}"
            detector.visualize(image, results, save_path)

            print(f"  检测到 {len(results)} 条鱼")
            for i, r in enumerate(results):
                print(f"    鱼 {i+1}: 置信度={r['score']:.3f}")
    else:
        print(f"测试目录不存在: {test_dir}")


if __name__ == '__main__':
    main()
