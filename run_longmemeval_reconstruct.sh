#!/usr/bin/env bash
set -euo pipefail

# Run LongMemEval reconstruct test with explicit parameters.
# Adjust values as needed.

LOG_FILE="logs_lme_reconstruct/lme_reconstruct_$(date +"%Y%m%d_%H%M%S").log"

python longmemeval_reconstruct_test.py \
  --data-dir data/longmemeval_s_items_reconstruct \
  --sample-use "5" \
  --memory-dir memory_longmem_eval \
  --chat-deployment "gpt-4o-mini" \
  --max-workers 3 \
  --use-image true \
  --use-profile none \
  --analyze-on wrong \
  --disable-response \
  | tee "$LOG_FILE"
