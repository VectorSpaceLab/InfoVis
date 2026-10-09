<h1 align="center">InfoVis: Unified Slide Image Generation and Editing</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-coming%20soon-b31b1b.svg" alt="paper">
  <img src="https://img.shields.io/badge/InfoVis%20Model-🤗%20coming%20soon-yellow" alt="model">
  <img src="https://img.shields.io/badge/VisSlideBench-🤗%20coming%20soon-yellow" alt="benchmark">
  <img src="https://img.shields.io/badge/Base-FLUX.2--klein--base--9B-blue" alt="base model">
</p>

<h4 align="center">
    <p>
        <a href=#-news>News</a> |
        <a href=#-introduction>Introduction</a> |
        <a href=#-quick-start>Quick Start</a> |
        <a href=#%EF%B8%8F-training>Training</a> |
        <a href=#%EF%B8%8F-citation>Citation</a>
    <p>
</h4>

<p align="center"><b>English</b> | <a href="README_zh.md">简体中文</a></p>

**InfoVis** is a unified 9B model for slide image generation and editing. One model handles text-to-slide generation, template-conditioned generation, multi-image grounded generation, and instruction-based slide editing. This repository contains the training code (continual pretraining and supervised fine-tuning) and the inference code.

## ✨ Highlights

- **Four slide tasks, one model**: All four tasks share one text-and-image interface, where the input images act as a style reference, content assets, or the slide to edit.
- **Slide-native data engine**: Over 13M continual-pretraining records and 1.9M supervised fine-tuning records, built from real presentations, executable PPTX edits, and synthetic slides. Instructions range from short requests to detailed descriptions and structured `style_json` specifications.
- **VisSlideBench**: 1,527 manually screened instances covering all four tasks, scored on eleven task-specific metrics. Its automatic rankings agree with blinded, dimension-wise human rankings.
- **Strong results**: InfoVis scores 7.94 overall on VisSlideBench, against 7.99 for Nano-Banana-2, and leads every evaluated open-source model on all eleven metrics. It is also the best open-source model on BizGenEval Slides, and our document-to-slide workflow built on it ranks second among ten systems in Aesthetics on SlidesGen-Bench.

## 🔥 News

- **2026-10-09**: We release the training code (continual pretraining + SFT) and the inference code of InfoVis.

## 📖 Introduction

Slides are a demanding target for image generation: a model has to render text accurately, arrange heterogeneous elements into a coherent layout, place supplied assets faithfully, and apply local or global edits while keeping the rest of the slide intact. InfoVis treats slide creation as conditional visual synthesis over a textual condition and optional input images:

| Task | Input images | Instruction |
|---|---|---|
| Text-to-slide | none | short request, detailed description, or `style_json` |
| Template-conditioned | one reference slide that sets the visual style | same three forms |
| Multi-image grounded | one to four assets that must appear on the slide | same three forms |
| Slide editing | the source slide | natural-language edit request |

**Data.** The slide data engine parses real presentations into text and localized visual assets and turns each page into text-to-slide, template-conditioned, and multi-image records that share the original slide as the target. Editing pairs come from executed PPTX edits, plus model-generated theme switching and image replacement for SFT.

<p align="center">
  <img src="assets/data_engine.png" width="95%">
  <br>
  <em>Slide data engine for continual pretraining (left) and supervised fine-tuning (right).</em>
</p>

**Training.** InfoVis keeps the FLUX.2-klein-base-9B architecture, raises the prompt length to 4,096 tokens, and trains only the diffusion transformer in three stages:

1. **Slide-native continual pretraining** on large-scale text-to-slide data.
2. **Four-task mixed continual pretraining**, extending the slide prior to style references, multi-image grounding, and editing.
3. **Task-specific SFT and model fusion**: a shared text-to-slide SFT checkpoint, four task-specific SFT branches started from it, and a linear merge of the four branches into the final model.

**VisSlideBench.** Generation is scored on Content Faithfulness (CF, OCR-based text fidelity plus input-image preservation for multi-image cases), Instruction Following (IF), and Aesthetic Quality (AQ). Editing is scored on Semantic Consistency (SC) and Presentation Quality (PQ). Judge-based dimensions use GPT-4.1.

<p align="center">
  <img src="assets/visslidebench_evaluation.png" width="95%">
  <br>
  <em>VisSlideBench curation and evaluation.</em>
</p>

**Main results on VisSlideBench.** Overall averages the eleven metrics and is reported only for models that support all four tasks.

| Model | T2I CF | T2I IF | T2I AQ | Template CF | Template IF | Template AQ | Multi CF | Multi IF | Multi AQ | Edit SC | Edit PQ | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Proprietary* | | | | | | | | | | | | |
| GPT-Image-2 | 8.52 | 8.57 | 7.71 | 8.19 | 8.51 | 7.62 | 8.93 | 8.44 | 7.61 | 9.80 | 8.04 | 8.36 |
| Nano-Banana-2 | 8.38 | 8.17 | 7.33 | 7.59 | 7.95 | 7.08 | 8.60 | 7.93 | 7.14 | 9.70 | 8.03 | 7.99 |
| Qwen-Image-2 | 7.18 | 6.89 | 6.38 | 8.23 | 6.29 | 5.74 | 6.97 | 6.25 | 5.55 | 4.96 | 6.47 | 6.45 |
| *Open-source* | | | | | | | | | | | | |
| FLUX.2-dev | 6.60 | 7.08 | 5.29 | 6.28 | 7.38 | 5.22 | 6.54 | 7.13 | 5.20 | 6.12 | 7.18 | 6.37 |
| FLUX.2-klein-base-9B | 6.29 | 7.11 | 5.19 | 6.44 | 7.24 | 5.40 | 6.57 | 6.99 | 5.32 | 5.98 | 6.72 | 6.30 |
| HiDream-O1-Image | 6.82 | 5.62 | 4.68 | 5.97 | 5.14 | 4.50 | 5.60 | 4.90 | 4.17 | 2.23 | 3.77 | 4.85 |
| BAGEL | 3.50 | 5.68 | 3.79 | 3.31 | 5.17 | 3.73 | 3.74 | 4.76 | 3.55 | 5.10 | 6.17 | 4.41 |
| OmniGen2 | 3.89 | 5.13 | 3.14 | 3.75 | 5.82 | 3.34 | 4.06 | 4.90 | 3.23 | 2.82 | 5.22 | 4.12 |
| **InfoVis (ours)** | **8.42** | **8.01** | **7.02** | **8.14** | **8.09** | **7.00** | **8.71** | **8.01** | **7.06** | **9.13** | **7.75** | **7.94** |

Single-task open-source models (Qwen-Image, Z-Image-Turbo, LongCat-Image, Qwen-Image-Edit, LongCat-Image-Edit) are compared on the tasks they support in the paper.

<p align="center">
  <img src="assets/edit_cases.jpg" width="80%">
  <br>
  <em>Slide editing with InfoVis (left: source, right: edited).</em>
</p>

## 📌 TODO

- [x] Training code (continual pretraining + SFT) and inference code.
- [ ] InfoVis model weights.
- [ ] VisSlideBench with its evaluation code.
- [ ] Training data.

## 🚀 Quick Start

### 🛠️ Environment Setup

```bash
# (Optional) Create a clean Python environment
conda create -n infovis python=3.12
conda activate infovis

# Install PyTorch for your CUDA version (CUDA 12.6 shown here)
pip install torch==2.7.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu126

# Clone this repository and install the remaining dependencies
git clone https://github.com/VectorSpaceLab/InfoVis.git
cd InfoVis
pip install -r requirements.txt
```

<details>
<summary>🌏 For users in Mainland China</summary>

```bash
# Install PyTorch from a domestic mirror
pip install torch==2.7.1 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126

# Install the other dependencies from a PyPI mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

InfoVis loads every module from `diffusers` and `transformers`; FLUX.2-klein support needs `diffusers>=0.37`. Download the base model (accept its license on Hugging Face first):

```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-base-9B
```

### 🧪 Inference

`--transformer` points to the InfoVis transformer (a diffusers-format folder). Pass input images in order: the reference slide for template-conditioned generation, one to four assets for multi-image grounded generation, or the source slide for editing.

```bash
# Text-to-slide
python inference.py --transformer checkpoints/InfoVis \
  --prompt "A clean title slide for a talk on renewable energy, with a wind turbine illustration on the left." \
  --out t2i.png

# Template-conditioned generation
python inference.py --transformer checkpoints/InfoVis \
  --prompt "Follow the style of the reference slide and create a slide titled 'Quarterly Results' ..." \
  --images reference.png --out template.png

# Multi-image grounded generation
python inference.py --transformer checkpoints/InfoVis \
  --prompt "Combine the two charts on a slide titled 'Experiment Results' with short captions." \
  --images chart1.png chart2.png --out multi_image.png

# Slide editing (use the source slide size)
python inference.py --transformer checkpoints/InfoVis \
  --prompt "Change the title color to red." \
  --images source.png --width 1280 --height 720 --out edited.png
```

The same function is available from Python:

```python
from inference import load_pipeline, generate, open_image

pipe = load_pipeline("black-forest-labs/FLUX.2-klein-base-9B", "checkpoints/InfoVis")
slide = generate(pipe, "Change the title color to red.", [open_image("source.png")], width=1280, height=720)
slide.save("edited.png")
```

Sampling uses 50 Euler steps, classifier-free guidance 4.0 with an empty negative prompt, a 4,096-token text context (longer prompts are truncated), and BF16. The default output size is 1376x768. Each input image is capped at 1024x1024 pixels and center-cropped to a multiple of 32.

### 📊 Run InfoVis on VisSlideBench

```bash
bash scripts/infer_visslidebench.sh checkpoints/InfoVis /path/to/VisSlideBench outputs/infovis
```

This generates all four tracks with one process per GPU and writes `outputs/infovis/<track>/<id>.png` with seed `2233 + index`. Text-to-slide cases are generated at 1280x720; template, multi-image, and editing cases use the `width` and `height` of their record. The scoring scripts ship with VisSlideBench.

## 🏋️ Training

### Data format

Each line of a training jsonl file is one sample. Image paths are relative to the data root.

```json
{"id": "sft/template/long/000000012", "task": "template",
 "instruction": "Match the attached slide's style ...",
 "input_images": ["images/sft/template/....png"],
 "output_image": "images/sft/....png",
 "target_img_size": [1376, 768]}
```

`configs/data/*.yaml` lists the jsonl sources of each stage. Point the configs at the data root and build the continual-pretraining replay subsets once:

```bash
export INFOVIS_DATA_ROOT=/path/to/InfoVis-Data
python tools/make_subsets.py --data_root $INFOVIS_DATA_ROOT
```

### Training stages

Every stage is full-parameter training of the transformer with FSDP (hybrid sharding), BF16, gradient checkpointing, and AdamW (lr 5e-5, betas 0.9/0.95, weight decay 0.01, gradient clipping 1.0), using one sample per GPU and eight gradient-accumulation steps. The text encoder (Qwen3) and the VAE stay frozen.

| Config | Stage | Initialized from | Nodes (8 GPUs) | Global batch | Steps |
|---|---|---|---:|---:|---:|
| `01_cpt_t2i` | Stage 1: text-to-slide CPT | FLUX.2-klein-base-9B | 6 | 384 | 13,500 |
| `02_cpt_mix_a` | Stage 2: four-task mixed CPT | `01` step 13,500 | 10 | 640 | 8,500 |
| `03_cpt_mix_b` | Stage 2: four-task mixed CPT | `02` step 8,500 | 10 | 640 | 6,000 |
| `04_cpt_mix_c` | Stage 2: four-task mixed CPT | `03` step 6,000 | 10 | 640 | 2,000 |
| `05_sft_t2i` | Stage 3: text-to-slide SFT | `04` step 2,000 | 10 | 640 | 4,500 |
| `06_sft_template` | Stage 3: template-conditioned SFT | `05` step 1,500 | 10 | 640 | 1,500 |
| `07_sft_multi_image` | Stage 3: multi-image grounded SFT | `05` step 1,500 | 12 | 768 | 2,500 |
| `08_sft_edit` | Stage 3: editing SFT | `05` step 1,500 | 12 | 768 | 500 |

Step 1,500 of `05_sft_t2i` is the shared checkpoint for the three other SFT branches. Launch a stage on all of its nodes:

```bash
NNODES=10 NODE_RANK=$RANK MASTER_ADDR=$MASTER bash scripts/train.sh configs/train/02_cpt_mix_a.yaml
```

Extra `key=value` arguments override the config (for example `train.max_steps=100`). A stopped run resumes from its latest checkpoint. With fewer GPUs, raise `train.gradient_accumulation_steps` so that `train.global_batch_size` stays the same.

### Export and merge

Each `checkpoint-N/` folder contains `pytorch_model_fsdp.bin`, the full transformer state dict. Export the four SFT branches and merge them linearly into the final model:

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

### Text features

The transformer reads three Qwen3 layers concatenated along the feature axis. `text_layer_offset` selects which tensors those are: `1` takes the outputs of decoder layers 9, 18, and 27, which all training stages used; `0` takes `hidden_states[9, 18, 27]`, the FLUX.2-klein pipeline convention, which our evaluation used. Training configs default to `1` and `inference.py` defaults to `0`. The two choices differ by one residual block; on training samples, the released model's flow-matching loss is the same under either (within 0.1%).

## 📂 Repository Structure

```
train.py                     one training script for every stage
inference.py                 single-case inference and VisSlideBench inference
infovis/modeling.py          model loading, Qwen3 text features, VAE latents
infovis/flow_matching.py     rectified-flow targets and timestep sampling
infovis/data.py              jsonl dataset, collator, resolution-bucket sampler
tools/export_transformer.py  training checkpoint -> diffusers transformer folder
tools/merge_checkpoints.py   linear merge of the four SFT transformers
tools/make_subsets.py        build CPT replay subsets from released id lists
configs/train/               the eight training stages
configs/data/                data mixture of each stage
configs/accelerate_fsdp.yaml FSDP settings
scripts/train.sh             multi-node training launcher
scripts/infer_visslidebench.sh  8-GPU VisSlideBench inference
```

## 📜 License

InfoVis is fine-tuned from FLUX.2-klein-base-9B. The model weights are a derivative of the FLUX model and are distributed under the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), for non-commercial use only.

## 🙏 Acknowledgements

InfoVis builds on [FLUX.2-klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) and [diffusers](https://github.com/huggingface/diffusers). Our training framework started from [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2), and the human validation of VisSlideBench follows the protocol of [EditScore](https://github.com/VectorSpaceLab/EditScore).

## ❤️ Citation

If you find this repository or our work useful, please consider giving a star ⭐ and citation 🦖:

```bibtex
@misc{infovis2026,
  title  = {InfoVis: Unified Slide Image Generation and Editing},
  year   = {2026}
}
```
