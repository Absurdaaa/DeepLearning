# Lab2 Name Classification Framework

这个目录现在已经整理成可复用的多文件实验框架，目标是完成作业要求里的：

- 原始 RNN 名字分类实验
- LSTM 名字分类实验
- `print(net)` 网络结构导出
- 验证集 `loss/accuracy` 曲线
- 验证集与测试集 confusion matrix

## 目录结构

```text
lab2/
├── train.py
├── src/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── engine.py
│   ├── models/
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── registry.py
│   └── utils/
│       ├── io.py
│       ├── paths.py
│       ├── plotting.py
│       └── runtime.py
├── docs/
├── rnn_pytorch_tutorial.ipynb
└── char_rnn_classification_tutorial.ipynb
```

## 数据放置方式

将名字分类数据按类别放到：

```text
lab2/data/names/
├── Arabic.txt
├── Chinese.txt
├── English.txt
└── ...
```

每个 `.txt` 文件对应一个类别，每行一个名字。

## 运行示例

原始 RNN：

```bash
python3 train.py --model rnn --epochs 30 --batch-size 64 --hidden-size 128
```

LSTM：

```bash
python3 train.py --model lstm --epochs 30 --batch-size 64 --hidden-size 128 --lr 1e-3
```

学习率扫描：

```bash
python3 sweep_lr.py --model rnn --optimizer adam --epochs 30 --batch-size 256 --hidden-size 128 --lrs 0.01 0.005 0.001 0.0005
python3 sweep_lr.py --model lstm --optimizer adam --epochs 30 --batch-size 256 --hidden-size 128 --lrs 0.01 0.005 0.001 0.0005
python3 sweep_lr.py --model myGRU --optimizer adam --epochs 30 --batch-size 256 --hidden-size 128 --lrs 0.01 0.005 0.001 0.0005
python3 sweep_lr.py --model myLSTM --optimizer adam --epochs 30 --batch-size 256 --hidden-size 128 --lrs 0.01 0.005 0.001 0.0005
```

## 输出内容

每次运行会在 `outputs/<model>/<run_name>/` 下生成：

- `model_structure.txt`
- `epoch_metrics.csv`
- `summary_metrics.csv`
- `run_metadata.json`
- `training_curves.png`
- `val_confusion_matrix.png`
- `test_confusion_matrix.png`
- `val_confusion_matrix.csv`
- `test_confusion_matrix.csv`
- `class_accuracy.csv`
- `best_model.pth`

学习率扫描还会在 `outputs/<model>/` 下额外生成：

- `<model>_<optimizer>_lr_sweep_summary.csv`
- `<model>_<optimizer>_best_lr.txt`

其中 `summary_metrics.csv` 会记录：

- `best_val_acc / best_val_loss / best_epoch`
- `test_acc / test_loss`
- `total_train_time_sec / avg_epoch_time_sec`
- `test_inference_time_sec / inference_time_per_batch_sec / inference_time_per_sample_ms`
- `param_count / trainable_param_count`
- `peak_memory_mb`
- `avg_test_sequence_length`
- `flops_per_sample`

## 说明

- `rnn` 是作业要求里的原始循环神经网络 baseline。
- `lstm` 是对应的改进模型，便于后续写“为什么 LSTM 性能优于 RNN”的分析。
- `myGRU` 和 `myLSTM` 是手动实现的门控循环网络，适合做加分项或结构理解。
- 当前框架默认做 `train/val/test` 三划分，适合直接写实验报告。
