"""Linear merge of task-specific transformers: W = sum_i c_i * W_i.

    python tools/merge_checkpoints.py \
        --checkpoints checkpoints/sft_t2i checkpoints/sft_template \
                      checkpoints/sft_multi_image checkpoints/sft_edit \
        --coefficients 0.25 0.25 0.25 0.25 --output_dir checkpoints/InfoVis

Each entry is a diffusers transformer folder or a pytorch_model_fsdp.bin file.
Weights are averaged in float32 and saved in bfloat16.
"""

import argparse
import os

import torch
from diffusers import Flux2Transformer2DModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="black-forest-labs/FLUX.2-klein-base-9B")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--coefficients", nargs="+", type=float, required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    assert len(args.checkpoints) == len(args.coefficients)
    if abs(sum(args.coefficients) - 1) > 1e-6:
        print(f"warning: coefficients sum to {sum(args.coefficients)}")

    merged = None
    for path, c in zip(args.checkpoints, args.coefficients):
        if os.path.isdir(path):
            sd = Flux2Transformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16).state_dict()
        else:
            sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if merged is None:
            merged = {k: c * v.float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                merged[k] += c * v.float()
        del sd

    model = Flux2Transformer2DModel.from_pretrained(args.base_model, subfolder="transformer", torch_dtype=torch.bfloat16)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in merged.items()})
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
