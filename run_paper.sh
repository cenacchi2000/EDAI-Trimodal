#!/usr/bin/env bash
set -e

python fast_multimodal.py \
  --edaic_root "data" \
  --daicwoz_root "data_daicwoz" \
  --metadata "metadata_mapped.csv" \
  --labels "Detailed_PHQ8_Labels.csv" \
  --max_participants 0 \
  --folds 7 \
  --rebuild_cache 0 \
  --use_gpu 0 \
  --outdir logs_paper \
  --seeds 1 2 3 4 5 6 7 8 9 10
