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

## 运行

```bash
python3 lab1/train.py --model simple_cnn --epochs 100 --batch-size 128
python3 lab1/train.py --model resnet18 --epochs 20 --batch-size 128
python3 lab1/train.py --model vgg11_bn --epochs 20 --batch-size 64
```

如果本地没有解压好的数据：

```bash
python3 lab1/train.py --download
```

## 输出

默认输出到：

```bash
lab1/outputs/<model_name>/
```

包括：

- `model_structure.txt`
- `train_samples.png`
- `training_curves.png`
- `val_predictions.png`
- `best_model.pth`
- `metrics.txt`

## 扩展模型

1. 在 `src/models/` 新建模型文件
2. 在 `src/models/registry.py` 里注册

例如：

```python
from .my_resnet import MyResNet

if model_name == "my_resnet":
    return MyResNet(num_classes=num_classes)
```
