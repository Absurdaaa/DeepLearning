#!/usr/bin/env bash

set -euo pipefail

cd /Users/linshangjin/Desktop/DeepLearning/lab2

python3 train_generation.py \
  --epochs 20 \
  --optimizer adam \
  --hidden-size 128 \
  --lr 0.001 \
  --dropout 0.1 \
  --clip-grad-norm 0 \
  --run-name generation_baseline

echo
echo "Finished. Check:"
echo "  outputs/generation/generation_baseline"
echo
echo "Key files:"
echo "  model_structure.txt"
echo "  epoch_metrics.csv"
echo "  summary_metrics.csv"
echo "  training_loss_curve.png"
echo "  generated_samples.txt"
