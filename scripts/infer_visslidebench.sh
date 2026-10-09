#!/bin/bash
# Generate all VisSlideBench predictions with one process per GPU.
#   bash scripts/infer_visslidebench.sh checkpoints/InfoVis /path/to/VisSlideBench outputs/infovis
# Then score with VisSlideBench/eval (see its README).
set -euo pipefail
cd "$(dirname "$0")/.."

TRANSFORMER=$1
BENCH=$2
OUT=$3
NUM_GPUS=${NUM_GPUS:-8}

for TASK in t2i template multi_image edit; do
  for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python inference.py --transformer "$TRANSFORMER" \
      --bench_root "$BENCH" --task "$TASK" --out_dir "$OUT" --shard "$i" --num_shards "$NUM_GPUS" &
  done
  wait
done
