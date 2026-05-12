# CIFAR-10 Training Project

入口：

```bash
python3 lab1/train.py --model simple_cnn --epochs 10
```

## 结构

```text
lab1/
├── train.py
├── src/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── engine.py
│   ├── models/
│   │   ├── registry.py
│   │   └── simple_cnn.py
│   └── utils/
│       ├── io.py
│       ├── plotting.py
│       └── runtime.py
├── data/
└── outputs/
```

## 命名说明

- `train.py`：入口脚本，避免 `train_cifar10.py` 这种重复命名
- `src/`：项目源码目录，替代过长的 `cifar10_framework/`
- `data.py`：数据加载和划分
- `engine.py`：训练、验证、测试主循环
- `models/`：模型定义和注册
- `utils/`：可视化、IO、运行环境工具
- `legacy/`：之前老师给的原始代码

## 运行

```bash
python3 lab1/train.py --model simple_cnn --optimizer sgd --save-plots
python3 lab1/train.py --model simple_cnn --optimizer sgd --epochs 10 --batch-size 128
python3 lab1/train.py --model resnet18 --optimizer adam --epochs 20 --batch-size 128
python3 lab1/train.py --model vgg11_bn --optimizer adamw --epochs 20 --batch-size 64
```

如果需要额外保存曲线图和预测图：

```bash
python3 lab1/train.py --save-plots
```

如果需要同步到 Weights & Biases：

```bash
python3 lab1/train.py --use-wandb --wandb-project cifar10-lab1
```

如果本地没有解压好的数据：

```bash
python3 lab1/train.py --download
```

优化器支持：

- `sgd`
- `adam`
- `adamw`

W&B 相关参数：

- `--use-wandb`
- `--wandb-project`
- `--wandb-entity`
- `--wandb-run-name`

## 输出

默认输出到：

```bash
lab1/outputs/<model_name>/
```

包括：

- `model_structure.txt`
- `train_samples.png`
- `epoch_metrics.csv`：每个 epoch 的 `train_loss`、`train_acc`、`val_loss`、`val_acc`
- `best_model.pth`
- `metrics.txt`
- `class_accuracy.txt`
- `class_accuracy.csv`

如果加了 `--save-plots`，还会额外保存：

- `training_curves.png`
- `val_predictions.png`

## 扩展模型

1. 在 `src/models/` 新建模型文件
2. 在 `src/models/registry.py` 里注册

例如：

```python
from .my_resnet import MyResNet

if model_name == "my_resnet":
    return MyResNet(num_classes=num_classes)
```
