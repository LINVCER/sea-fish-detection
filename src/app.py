"""
Flask 前端 - 水下鱼类检测系统
"""
import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from config import Config
from model import FasterRCNNModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 限制

# 全局配置
config = Config()
UPLOAD_DIR = Path("static/uploads")
RESULT_DIR = Path("static/results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 加载模型
device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
model = FasterRCNNModel(num_classes=config.NUM_CLASSES, pretrained=False)
checkpoint = torch.load(config.MODEL_DIR / "best_model.pth", map_location=device)
model.model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"模型加载完成, 设备: {device}")


def detect_image(image_path, confidence_threshold=0.5):
    """检测单张图像"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []

    # 预处理
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(pil_image).to(device)

    # 推理
    with torch.no_grad():
        preds = model.model([tensor])[0]

    # 后处理
    results = []
    boxes = preds['boxes'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] >= confidence_threshold:
            results.append({
                'box': boxes[i].astype(int).tolist(),
                'score': float(scores[i]),
                'label': int(labels[i])
            })

    return image, results


def draw_results(image, results):
    """绘制检测结果"""
    vis_image = image.copy()
    colors = {1: (0, 255, 0)}

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

    # 统计信息
    num_fish = len(results)
    avg_score = np.mean([r['score'] for r in results]) if results else 0
    info_text = f"Fish: {num_fish}, Avg Score: {avg_score:.2f}"
    cv2.putText(vis_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis_image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    # 保存上传文件
    file_ext = Path(file.filename).suffix or '.jpg'
    filename = f"{uuid.uuid4().hex}{file_ext}"
    upload_path = UPLOAD_DIR / filename
    file.save(str(upload_path))
    print(f"上传文件保存至: {upload_path}")

    # 检测
    confidence = float(request.form.get('confidence', 0.5))
    image, results = detect_image(upload_path, confidence)

    if image is None:
        return jsonify({'error': '无法读取图像'}), 400

    print(f"检测到 {len(results)} 条鱼")

    # 绘制结果
    result_image = draw_results(image, results)
    result_filename = f"result_{filename}"
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), result_image)
    print(f"结果保存至: {result_path}")

    return jsonify({
        'success': True,
        'result_image': f"/static/results/{result_filename}",
        'fish_count': len(results),
        'detections': results,
        'average_score': float(np.mean([r['score'] for r in results])) if results else 0
    })


@app.route('/static/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(str(RESULT_DIR), filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
