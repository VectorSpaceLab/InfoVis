"""Convert a training checkpoint into a diffusers transformer folder for inference.

    python tools/export_transformer.py \
        --checkpoint experiments/05_sft_t2i/checkpoint-4500/pytorch_model_fsdp.bin \
        --output_dir checkpoints/sft_t2i
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infovis.modeling import load_transformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="black-forest-labs/FLUX.2-klein-base-9B")
    ap.add_argument("--checkpoint", required=True, help="pytorch_model_fsdp.bin from a training run")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    model = load_transformer(args.base_model, args.checkpoint)
    model.to(torch.bfloat16).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
