# Training InfoVis

InfoVis trains only the FLUX.2-klein diffusion transformer; the Qwen3 text encoder and the VAE stay frozen. Training has three stages: text-to-slide continual pretraining (CPT), four-task mixed CPT, and task-specific supervised fine-tuning (SFT) followed by a linear merge.

## Data

Each line of a training jsonl file is one sample. Image paths are relative to the data root.

```json
{"id": "sft/template/long/000000012", "task": "template",
 "instruction": "Match the attached slide's style ...",
 "input_images": ["images/sft/template/....png"],
 "output_image": "images/sft/....png",
 "target_img_size": [1376, 768]}
```

`configs/data/*.yaml` lists the jsonl files of each stage. Set the data root and build the CPT replay subsets once:

```bash
export INFOVIS_DATA_ROOT=/path/to/InfoVis-Data
python tools/make_subsets.py --data_root $INFOVIS_DATA_ROOT
```

## Stages

All stages use FSDP (hybrid sharding), BF16, gradient checkpointing, and AdamW (lr 5e-5, betas 0.9/0.95, weight decay 0.01, gradient clipping 1.0), with one sample per GPU and eight gradient-accumulation steps.

| Config | Stage | Initialized from | Nodes (8 GPUs) | Global batch | Steps |
|---|---|---|---:|---:|---:|
| `01_cpt_t2i` | Text-to-slide CPT | FLUX.2-klein-base-9B | 6 | 384 | 13,500 |
| `02_cpt_mix_a` | Four-task mixed CPT | `01` step 13,500 | 10 | 640 | 8,500 |
| `03_cpt_mix_b` | Four-task mixed CPT | `02` step 8,500 | 10 | 640 | 6,000 |
| `04_cpt_mix_c` | Four-task mixed CPT | `03` step 6,000 | 10 | 640 | 2,000 |
| `05_sft_t2i` | Text-to-slide SFT | `04` step 2,000 | 10 | 640 | 4,500 |
| `06_sft_template` | Template-conditioned SFT | `05` step 1,500 | 10 | 640 | 1,500 |
| `07_sft_multi_image` | Multi-image grounded SFT | `05` step 1,500 | 12 | 768 | 2,500 |
| `08_sft_edit` | Editing SFT | `05` step 1,500 | 12 | 768 | 500 |

Step 1,500 of `05_sft_t2i` is the shared starting point of the other three SFT branches. Run each stage on all of its nodes:

```bash
NNODES=10 NODE_RANK=$RANK MASTER_ADDR=$MASTER bash scripts/train.sh configs/train/02_cpt_mix_a.yaml
```

Extra `key=value` arguments override the config (for example `train.max_steps=100`). A stopped run resumes from its latest checkpoint. With fewer GPUs, raise `train.gradient_accumulation_steps` to keep `train.global_batch_size`.

## Export and merge

Each `checkpoint-N/` folder holds `pytorch_model_fsdp.bin`, the full transformer state dict. Export the four SFT branches and merge them:

```bash
python tools/export_transformer.py --checkpoint experiments/05_sft_t2i/checkpoint-4500/pytorch_model_fsdp.bin --output_dir checkpoints/sft_t2i
python tools/export_transformer.py --checkpoint experiments/06_sft_template/checkpoint-1500/pytorch_model_fsdp.bin --output_dir checkpoints/sft_template
python tools/export_transformer.py --checkpoint experiments/07_sft_multi_image/checkpoint-2500/pytorch_model_fsdp.bin --output_dir checkpoints/sft_multi_image
python tools/export_transformer.py --checkpoint experiments/08_sft_edit/checkpoint-500/pytorch_model_fsdp.bin --output_dir checkpoints/sft_edit

python tools/merge_checkpoints.py \
  --checkpoints checkpoints/sft_t2i checkpoints/sft_template checkpoints/sft_multi_image checkpoints/sft_edit \
  --coefficients $W_T2I $W_TEMPLATE $W_MULTI $W_EDIT --output_dir checkpoints/InfoVis
```

Merging runs on CPU and needs about 100 GB of RAM.

## Notes

- **Text features.** The transformer reads three Qwen3 layers concatenated along the feature axis. `text_layer_offset: 1` (training default) takes the outputs of decoder layers 9, 18, and 27; `--text_layer_offset 0` (inference default) takes `hidden_states[9, 18, 27]`, the FLUX.2-klein pipeline convention. They differ by one residual block, and the released model's flow-matching loss on training samples is the same under either (within 0.1%).
- **Sampling.** 50 Euler steps, classifier-free guidance 4.0 with an empty negative prompt, a 4,096-token text context, and BF16. Input images are capped at 1024x1024 pixels and center-cropped to multiples of 32.
- **VisSlideBench resolution.** `scripts/infer_visslidebench.sh` generates text-to-slide cases at 1280x720, template and multi-image cases at 1376x768, and editing cases at the size of the source slide (`width` x `height` of the record), with seed `2233 + index`.
