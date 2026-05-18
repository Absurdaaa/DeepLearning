#!/usr/bin/env bash

set -euo pipefail

cd /Users/linshangjin/Desktop/DeepLearning/lab3

# 当前脚本只保留本实验真正要跑的学习率扫描命令。
# 等扫描完成后，再根据 best_lr.txt 选择正式训练配置并生成报告素材。

###############################################################################
# 一、必做部分
###############################################################################

# 纯 RNN Seq2Seq baseline
python3 sweep_lr.py \
  --model seq2seq_rnn \
  --optimizer adam \
  --epochs 30 \
  --batch-size 64 \
  --hidden-size 128 \
  --teacher-forcing-ratio 0.5 \
  --max-samples 12000 \
  --lrs 0.01 0.005 0.003 0.001

# 注意力 Seq2Seq
python3 sweep_lr.py \
  --model seq2seq_attn \
  --optimizer adam \
  --epochs 30 \
  --batch-size 64 \
  --hidden-size 128 \
  --teacher-forcing-ratio 0.5 \
  --max-samples 12000 \
  --lrs 0.01 0.005 0.003 0.001

###############################################################################
# 二、运行结束后建议执行
###############################################################################

# python3 scripts/generate_report_assets.py
