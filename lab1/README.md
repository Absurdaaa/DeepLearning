# CIFAR-10 Training Project

入口：

```bash
python3 train.py --model simple_cnn --epochs 10
```

## 结构

```text

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
python3 train.py --model simple_cnn --optimizer sgd --save-plots
python3 train.py --model simple_cnn --optimizer sgd --epochs 10 --batch-size 512
python3 train.py --model resnet20 --optimizer sgd --lr 0.05 --run-name baseline
python3 train.py --model densenet_bc_100 --optimizer sgd --lr 0.05
python3 train.py --model mobilenet_v1 --optimizer sgd --lr 0.05
python3 train.py --model vgg11_bn --optimizer adamw --lr 1e-3
```

如果你要固定 `bs=512` 批量扫学习率：

```bash
python3 sweep_lr.py --model simple_cnn --optimizer sgd --batch-size 512 --epochs 100 --lrs 0.2 0.1 0.05 0.02 0.01 0.005

python3 sweep_lr.py --model resnet20 --optimizer sgd --batch-size 512 --epochs 100 --lrs 0.2 0.1 0.05 0.02 0.01

python3 sweep_lr.py --model vgg11_bn --optimizer adamw --batch-size 512 --epochs 100 --lrs 0.01 0.005 0.001 0.0005 0.0001
```

如果显存够，可以并行扫：

```bash
python3 lab1/sweep_lr.py --model simple_cnn --optimizer sgd --batch-size 512 --epochs 100 --lrs 0.2 0.1 0.05 0.02 0.01 


```

如果有多张卡，也可以指定设备列表：

```bash
python sweep_lr.py --model densenet_bc_100 --optimizer sgd --batch-size 512 --epochs 100 --lrs 0.2 0.1 0.05 0.02 0.01 0.005 --max-parallel 2 --devices cuda:0 cuda:1

python sweep_lr.py --model res2net29_8c64w --optimizer sgd --batch-size 512 --epochs 100 --lrs 0.2 0.1 0.05 0.02 0.01 0.005 --max-parallel 1 --devices cuda:0 cuda:1
```

如果需要额外保存曲线图和预测图：

```bash
python3 train.py --save-plots
```

如果需要同步到 Weights & Biases：

```bash
python3 train.py --use-wandb --wandb-project cifar10-lab1
```

如果本地没有解压好的数据：

```bash
python3 train.py --download
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
outputs/<model_name>/<run_name>/
```

包括：

- `model_structure.txt`
- `train_samples.png`
- `epoch_metrics.csv`：每个 epoch 的 `train_loss`、`train_acc`、`val_loss`、`val_acc`、单轮耗时、累计训练耗时
- `summary_metrics.csv`：参数量、FLOPs、总训练时间、推理时间、收敛轮次、准确率等汇总指标
- `best_model.pth`
- `metrics.txt`
- `class_accuracy.txt`
- `class_accuracy.csv`

如果加了 `--save-plots`，还会额外保存：

- `training_curves.png`
- `val_predictions.png`

学习率扫描脚本还会额外输出：

- `<sweep_name>_lr_sweep_summary.csv`
- `<sweep_name>_best_lr.txt`

重复执行同一条 `sweep_lr.py` 命令时，如果某个学习率对应目录下已经存在 `summary_metrics.csv`，脚本会自动跳过该学习率，只继续未完成的部分。

## 扩展模型

1. 在 `src/models/` 新建模型文件
2. 在 `src/models/registry.py` 里注册

例如：

```python
from .my_resnet import MyResNet

if model_name == "my_resnet":
    return MyResNet(num_classes=num_classes)
```
