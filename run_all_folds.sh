#!/usr/bin/env bash
set -e
LOG="outputs/all_folds_eval.log"
mkdir -p outputs
for fold in $(seq 0 27); do
    echo "===== START FOLD $fold =====" | tee -a "$LOG"
    python scripts/eval_test_split.py \
        --checkpoint-pattern "checkpoints/mphoi72_fold{fold}/final-model/best_model.pth" \
        --num-folds 28 \
        --fold "$fold" \
        --output-dir "outputs/test_eval_fold${fold}" \
        --dataset-root data/mphoi72 \
        2>&1 | tee -a "$LOG"
    echo "===== DONE FOLD $fold =====" | tee -a "$LOG"
done
echo "ALL FOLDS COMPLETE" | tee -a "$LOG"
