#!/bin/bash
# Launch one training stage on this node. Run the same command on every node.
#   NNODES=10 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train.sh configs/train/02_cpt_mix_a.yaml
# The node count must match the config: global_batch_size = NNODES * 8 * batch_size * gradient_accumulation_steps.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=$1; shift
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

accelerate launch --config_file configs/accelerate_fsdp.yaml \
  --num_machines "$NNODES" --num_processes $((NNODES * GPUS_PER_NODE)) \
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  train.py --config "$CONFIG" "$@"
