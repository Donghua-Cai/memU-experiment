#!/usr/bin/env bash
set -euo pipefail

# Batch-evaluate specific LongMemEval samples with explicit parameters.
# Adjust values as needed.

LOG_FILE="longmemeval_eval_single_$(date +"%Y%m%d_%H%M%S").log"

SAMPLES=(
  "00001_e47becba"
  "00002_118b2229"
  "00003_51a45a95"
  "00004_58bf7951"
  "00005_1e043500"
)

for sample in "${SAMPLES[@]}"; do
  echo "=== Evaluating ${sample} ==="
  python longmemeval_eval_single.py \
    --sample-file "data/longmemeval_s_items/${sample}.json" \
    --memory-dir memory_longmem_eval/longmemeval_reconstruct \
    --chat-deployment "gpt-4o-mini" \
    --use-profile none
  echo ""
done | tee "$LOG_FILE"
