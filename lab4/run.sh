#!/usr/bin/env bash

set -euo pipefail


# 这是 lab4 的总控脚本。
# 默认按“先扫学习率 -> 再正式训练 -> 最后整理报告素材”的顺序组织。
#
# 本实验的核心任务：
# - 训练基础版 GAN（FashionMNIST）
# - 训练卷积版 DCGAN（FashionMNIST）
# - 导出 G / D loss 曲线与模型结构
# - 生成 8 张固定噪声样例图
# - 生成 5 组潜变量扰动分析图（每组 3 次调整，共 15 x 8 张）
#
# 当前推荐流程：
# 1. 分别为 gan / dcgan 扫学习率
# 2. 用最佳学习率做正式训练
# 3. 汇总报告所需图表与表格
#
# 说明：
# - 下面命令按 lab4 预期接口先写好，等 train.py / sweep_lr.py / 报告脚本补齐后可直接使用
# - 如果你只想执行某一步，也可以手动复制对应命令单独运行


echo "== [1/5] Sweep GAN learning rates =="
python3 sweep_lr.py \
  --model gan \
  --optimizer adam \
  --epochs 50 \
  --batch-size 128 \
  --latent-dim 100 \
  --lrs 0.001 0.0005 0.0002 0.0001


echo "== [2/5] Sweep DCGAN learning rates =="
python3 sweep_lr.py \
  --model dcgan \
  --optimizer adam \
  --epochs 50 \
  --batch-size 128 \
  --latent-dim 100 \
  --lrs 0.001 0.0005 0.0002 0.0001


echo "== [3/5] Train final GAN =="
python3 train.py \
  --model gan \
  --run-name final_gan_fashionmnist \
  --epochs 100 \
  --batch-size 128 \
  --latent-dim 100 \
  --optimizer adam \
  --lr 0.0002


echo "== [4/5] Train final DCGAN =="
python3 train.py \
  --model dcgan \
  --run-name final_dcgan_fashionmnist \
  --epochs 100 \
  --batch-size 128 \
  --latent-dim 100 \
  --optimizer adam \
  --lr 0.0002


echo "== [5/5] Generate report assets =="
python3 scripts/generate_report_assets.py \
  --gan-run final_gan_fashionmnist \
  --dcgan-run final_dcgan_fashionmnist \
  --fixed-sample-count 8 \
  --latent-analysis-count 100 \
  --latent-analysis-picks 5 \
  --latent-perturbations 3


echo "All lab4 stages completed."
