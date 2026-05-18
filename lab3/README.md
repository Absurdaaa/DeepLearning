# Lab3 Seq2Seq Translation Framework

本目录已按 `LAB_PROJECT_STRUCTURE.md` 规范整理成可复现的多文件实验框架，用于完成：

- 基于纯 RNN 解码器的 Seq2Seq 翻译实验
- 基于注意力机制的 Seq2Seq 翻译实验
- 学习率扫描、正式训练输出、报告素材生成

## 目录结构

```text
lab3/
├── README.md
├── docs/
├── data/
│   └── eng-fra.txt
├── outputs/
├── run.sh
├── sweep_lr.py
├── train.py
├── scripts/
│   └── generate_report_assets.py
├── src/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── engine.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── registry.py
│   │   ├── seq2seq_attn.py
│   │   └── seq2seq_rnn.py
│   └── utils/
│       ├── io.py
│       ├── paths.py
│       ├── plotting.py
│       ├── profiling.py
│       └── runtime.py
├── old/
│   ├── seq2seq.zip
│   └── seq2seq_tutorial/
└── 实验模板/
```

## 模型说明

- `seq2seq_rnn`
  - baseline，编码器 + 无注意力 GRU 解码器
- `seq2seq_attn`
  - Bahdanau attention 解码器

默认数据流程沿用官方教程的英法平行语料，并按教程习惯做：

- 英法句对读取与标准化
- 句长过滤
- 英语前缀过滤
- 默认做 `fra -> eng` 翻译任务

## 运行示例

单次训练：

```bash
python3 train.py --model seq2seq_rnn --epochs 30 --batch-size 64 --hidden-size 128 --lr 0.003
python3 train.py --model seq2seq_attn --epochs 30 --batch-size 64 --hidden-size 128 --lr 0.003
```

学习率扫描：

```bash
python3 sweep_lr.py --model seq2seq_rnn --optimizer adam --epochs 30 --batch-size 64 --hidden-size 128 --lrs 0.01 0.005 0.003 0.001
python3 sweep_lr.py --model seq2seq_attn --optimizer adam --epochs 30 --batch-size 64 --hidden-size 128 --lrs 0.01 0.005 0.003 0.001
```

报告素材生成：

```bash
python3 scripts/generate_report_assets.py
```

## 输出说明

正式训练输出保存在：

```text
outputs/<model>/<run_name>/
```

每个 run 至少包含：

- `model_structure.txt`
- `epoch_metrics.csv`
- `summary_metrics.csv`
- `run_metadata.json`
- `best_model.pth`
- `training_curves.png`
- `sample_translations.csv`

注意力模型还会额外输出：

- `attention_examples/*.png`

学习率扫描结果保存在：

- `outputs/<model>/<model>_<optimizer>_lr_sweep_summary.csv`
- `outputs/<model>/<model>_<optimizer>_best_lr.txt`

报告素材脚本会生成：

- `实验模板/fig/generated/*.png`
- `实验模板/tables/*.tex`
- `实验模板/generated_assets_manifest.txt`
