# DeepLearning Lab Project Structure

这份文件用于约束 `lab3 / lab4 / 后续实验` 的项目结构与输出规范。  
以后新实验开始时，可以直接把这份文件发给 agent，减少重复说明。

## 目标

后续实验项目默认整理成和 `lab1`、`lab2` 风格一致的多文件结构，满足：

- 代码清晰可维护
- 实验可重复运行
- 输出结果可直接用于写实验报告
- 后续 agent 不需要重新猜目录结构和文件职责

## 顶层目录规范

每个实验目录默认采用如下结构：

```text
labX/
├── README.md
├── docs/
│   └── 要求.md
├── data/
├── outputs/
├── train.py
├── sweep_lr.py
├── run.sh
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── engine.py
│   ├── models/
│   └── utils/
├── old/                  # 老教程、原始 notebook、参考代码
└── 实验模板/              # LaTeX 模板、图片、参考文献
```

如果当前实验不需要某一项，可以不创建，但优先保持这个骨架。

## 文件职责

### 入口文件

- `train.py`
  - 单次正式训练入口
  - 负责读取配置、构建数据、构建模型、启动训练、保存结果

- `sweep_lr.py`
  - 学习率扫描入口
  - 对每个模型单独扫学习率
  - 自动跳过已经完成的 run
  - 输出学习率汇总表和最佳学习率文本

- `run.sh`
  - 批量运行脚本
  - 默认只写“当前实验真正要跑的命令”
  - 如果用户要求统一扫学习率，就不要在这里混入正式训练命令

### `src/` 内部职责

- `config.py`
  - 只负责 CLI 参数和配置 dataclass

- `constants.py`
  - 公共常量，例如字符表、类别名、默认尺寸等

- `data.py`
  - 数据集读取
  - 数据预处理
  - train/val/test 划分
  - dataloader / collate_fn

- `engine.py`
  - 训练循环
  - 验证循环
  - 测试循环
  - checkpoint 保存
  - 指标汇总

- `models/`
  - 每个模型单独一个文件
  - `registry.py` 统一注册模型名

- `utils/`
  - `io.py`：写 CSV / JSON / txt
  - `plotting.py`：曲线图、混淆矩阵、类别准确率图
  - `profiling.py`：参数量、FLOPs、时间、显存
  - `runtime.py`：seed、matplotlib、环境设置
  - `paths.py`：run_name、输出路径规则

## 模型文件规范

`src/models/` 默认遵循：

```text
src/models/
├── __init__.py
├── registry.py
├── rnn.py
├── lstm.py
├── myGRU.py
└── myLSTM.py
```

要求：

- 每个模型文件只放一个主模型类
- 文件名尽量和 `--model` 名称保持一致
- 手写模型和框架内置模型分开
- `registry.py` 里统一维护：
  - `AVAILABLE_MODELS`
  - `build_model(...)`

## 输出目录规范

每次正式训练都输出到：

```text
outputs/<model>/<run_name>/
```

至少包含：

- `model_structure.txt`
- `epoch_metrics.csv`
- `summary_metrics.csv`
- `run_metadata.json`
- `best_model.pth`

如果实验需要图：

- `training_curves.png`
- `val_confusion_matrix.png`
- `test_confusion_matrix.png`
- `class_accuracy.csv`
- 其他补充图

## CSV/日志规范

### `epoch_metrics.csv`

建议至少记录：

- `epoch`
- `train_loss`
- `train_acc`
- `val_loss`
- `val_acc`
- `train_val_acc_gap`
- `epoch_time_sec`
- `elapsed_train_time_sec`

### `summary_metrics.csv`

建议至少记录：

- `best_val_acc`
- `best_val_loss`
- `best_epoch`
- `test_acc`
- `test_loss`
- `param_count`
- `trainable_param_count`
- `flops_per_sample` 或 `flops_per_image`
- `total_train_time_sec`
- `avg_epoch_time_sec`
- `test_inference_time_sec`
- `inference_time_per_sample_ms`
- `peak_memory_mb`

### `run_metadata.json`

建议记录：

- `model`
- `run_name`
- `epochs`
- `batch_size`
- `optimizer`
- `lr`
- `hidden_size`
- `num_layers`
- `dropout`
- `seed`
- `device`
- `train_size`
- `val_size`
- `test_size`
- `class_count`
- `class_names`

## 学习率扫描规范

默认原则：

- 每个模型都单独扫学习率
- `batch_size` 在 sweep 和正式实验中尽量统一
- 最终以 `best_val_acc` 选学习率
- 测试集只用于最后报告，不用于选超参数

`sweep_lr.py` 需要输出：

```text
outputs/<model>/<model>_<optimizer>_lr_sweep_summary.csv
outputs/<model>/<model>_<optimizer>_best_lr.txt
```

## 注释规范

要求：

- 优先写简洁中文注释
- 只在关键位置加注释，不要整文件全是注释
- 变量名、函数名、PyTorch API 保持英文

重点注释位置：

- 数据编码逻辑
- 变长序列处理
- 模型前向关键公式
- 梯度裁剪 / BPTT / checkpoint 选择
- FLOPs 和时间统计逻辑

## `old/` 目录规范

`old/` 用于保留：

- 老教程
- 原始 notebook
- 老版本参考代码

如果需要把 notebook 转成 `.py`：

- 保留原 `.ipynb`
- 生成同名 `.py`
- 说明性文字尽量中文化
- 变量名和 API 不翻译

## `实验模板/` 目录规范

用于放报告相关内容：

- `main.tex`
- `reference.bib`
- `fig/`
- `tables/`
- style 文件

要求：

- 实验图表尽量从 `outputs/` 自动生成
- 生成的图、表尽量落到 `实验模板/fig`、`实验模板/tables`
- 最终 PDF 路径固定清晰

## README 规范

每个实验目录都应有 `README.md`，至少写清楚：

- 任务是什么
- 数据放在哪
- 怎么运行 `train.py`
- 怎么运行 `sweep_lr.py`
- 输出目录里有什么
- 哪些模型是必做，哪些是加分项

## 给 agent 的默认要求

以后你可以直接把下面这段发给 agent：

```text
请按 /Users/linshangjin/Desktop/DeepLearning/LAB_PROJECT_STRUCTURE.md 的规范组织当前实验目录。优先保持和 lab1、lab2 一致的多文件结构，不要堆成单文件 notebook。训练输出、学习率扫描、报告素材都按这份规范生成。
```

## 特别说明

- 默认先做 Markdown 规范，不强制做 Codex skill
- 如果后面你确认这份规范长期稳定，再把它升级成真正的 skill 更合适
- 在规范未变之前，以这份文件为最高优先级的项目结构说明
