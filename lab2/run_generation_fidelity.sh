#!/usr/bin/env bash

set -euo pipefail

# cd /Users/linshangjin/Desktop/DeepLearning/lab2

# 默认用当前分类任务里最强的 LSTM 判别器做“评委”
BEST_INFO_FILE="outputs/lstm/lstm_adam_best_lr.txt"
if [[ ! -f "${BEST_INFO_FILE}" ]]; then
  echo "Missing ${BEST_INFO_FILE}. Please run classification sweep first."
  exit 1
fi

CLASSIFIER_RUN_NAME="$(grep '^run_name=' "${BEST_INFO_FILE}" | cut -d'=' -f2-)"
CLASSIFIER_RUN_DIR="outputs/lstm/${CLASSIFIER_RUN_NAME}"

GEN_RUNS=(
  "outputs/generation/rnn_gen_baseline"
  "outputs/generation/lstm_gen_baseline"
  "outputs/generation/gru_gen_baseline"
)

for RUN_DIR in "${GEN_RUNS[@]}"; do
  if [[ ! -f "${RUN_DIR}/generated_samples.txt" ]]; then
    echo "Missing generated samples: ${RUN_DIR}/generated_samples.txt"
    echo "Please run bash run_generation.sh first."
    exit 1
  fi
done

python3 evaluate_generation_fidelity.py \
  --classifier-run "${CLASSIFIER_RUN_DIR}" \
  --generation-runs "${GEN_RUNS[@]}" \
  --report-name "baseline_lstm_judge"

echo
echo "Finished generation fidelity evaluation. Check:"
echo "  outputs/generation/fidelity_reports/baseline_lstm_judge"
echo
echo "Key files:"
echo "  fidelity_summary.csv"
echo "  overall_fidelity.png"
echo "  category_fidelity.png"
echo "  *_fidelity_confusion_matrix.png"
