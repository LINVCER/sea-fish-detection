# 水下鱼类目标检测系统

基于深度学习的水下鱼类目标检测与分割系统，应用于海洋生态监测和渔业资源管理。

## 项目概述

本项目采用 Faster R-CNN 深度学习模型，实现水下鱼类的自动检测与识别。通过训练 DeepFish 数据集，模型能够准确识别多种海洋鱼类，为海洋生态保护和渔业资源管理提供技术支持。

### 主要成果

- **mAP@50**: 96.46%
- **Recall**: 96.99%
- **F1-Score**: 87.94%

## 项目结构

```
sea_detection_project/
├── src/                        # 源代码
│   ├── config.py              # 配置文件
│   ├── dataset.py             # 数据集加载
│   ├── model.py               # 模型定义
│   ├── train.py               # 训练脚本
│   ├── inference.py           # 推理脚本
│   ├── evaluate.py            # 评估脚本
│   ├── generate_figures.py    # 图表生成
│   ├── app.py                 # Web应用后端
│   ├── templates/             # HTML模板
│   │   └── index.html         # 前端页面
│   └── static/                # 静态资源
│       ├── uploads/           # 上传目录
│       └── results/           # 结果目录
│
├── data/                       # 数据目录
│   └── processed/             # 处理后的数据
│
├── models/                     # 模型权重
│
├── outputs/                    # 输出结果
│   └── visualizations/        # 可视化结果
│
├── figures/                    # 论文图表
│   ├── 图1_训练损失曲线.png
│   ├── 图2_mAP变化曲线.png
│   ├── 图3_学习率调度曲线.png
│   ├── 图4_测试集评估结果.png
│   ├── 图5_检测统计饼图.png
│   ├── 图6_训练过程综合分析.png
│   ├── 图7_模型架构示意图.png
│   └── 图8_各类别检测结果.png
│
├── docs/                       # 文档
│   ├── report_full.md         # 完整报告
│   ├── evaluation_results.txt # 评估结果
│   └── training_log.txt       # 训练日志
│
├── requirements.txt            # 依赖列表
└── README.md                  # 项目说明
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- Python 3.10+
- PyTorch 2.0+
- torchvision
- opencv-python
- matplotlib
- numpy

## 使用方法

### 1. 数据准备

```bash
# 下载数据集（需要 Kaggle API）
python src/config_kaggle.py

# 准备数据集
python src/prepare_dataset.py
```

### 2. 训练模型

```bash
python src/train.py
```

### 3. 推理检测

```bash
python src/inference.py
```

### 4. 评估模型

```bash
python src/evaluate.py
```

### 5. 生成图表

```bash
python src/generate_figures.py
```

### 6. 启动Web应用

```bash
python src/app.py
```

然后在浏览器中访问: http://localhost:5000

Web应用功能：
- 上传水下图像
- 实时检测鱼类
- 显示检测结果和置信度
- 可视化边界框

## 模型架构

本项目采用 Faster R-CNN (ResNet-50-FPN) 作为基础架构：

- **Backbone**: ResNet-50 + FPN
- **检测头**: Fast R-CNN
- **输入尺寸**: 800 × 800
- **输出**: 边界框 + 类别

## 训练配置

| 参数 | 值 |
|------|-----|
| Batch Size | 1 |
| Learning Rate | 0.005 |
| Epochs | 50 |
| Optimizer | Adam |
| Weight Decay | 0.0005 |
| AMP | 开启 |

## 评估结果

### 定量评估

| 指标 | 数值 |
|------|------|
| mAP@50 | 96.46% |
| mAP@75 | 60.75% |
| mAP@50:95 | 57.10% |
| Precision | 80.43% |
| Recall | 96.99% |
| F1-Score | 87.94% |

### 检测统计

| 统计项 | 数量 |
|--------|------|
| 正确检测 (TP) | 1,611 |
| 误检 (FP) | 392 |
| 漏检 (FN) | 50 |

## 数据集

本项目使用 [DeepFish](https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection) 数据集：

- **图像总数**: 4,505 张
- **鱼类种类**: 20 种
- **标注格式**: YOLO 格式
- **训练集**: 3,604 张 (80%)
- **验证集**: 450 张 (10%)
- **测试集**: 451 张 (10%)

## 训练曲线

### 损失曲线

训练损失从 0.3316 下降至 0.2105，下降幅度 36.5%。

### mAP 曲线

mAP@50 从 90.47% 提升至 96.46%，提升 5.99 个百分点。

## 应用场景

1. **海洋生态监测**: 自动识别鱼类种群，监测生态变化
2. **渔业资源管理**: 估算渔获量，辅助管理决策
3. **水下机器人导航**: 环境感知与目标识别

## 未来优化

1. 引入注意力机制，提升检测精度
2. 研究模型轻量化，实现实时检测
3. 扩展数据集，增加更多鱼类种类
4. 结合视频序列，实现目标跟踪

## 许可证

MIT License

## 参考文献

- He, K., et al. "Faster R-CNN." IEEE TPAMI, 2017.
- He, K., et al. "Mask R-CNN." ICCV, 2017.
- Lin, T.Y., et al. "Feature Pyramid Networks." CVPR, 2017.

## 联系方式

如有问题，请提交 Issue 或联系作者。
