<h1 align="center">InfoVis: Unified Slide Image Generation and Editing</h1>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv%20paper-coming%20soon-b31b1b.svg" alt="arxiv">
  <a href="https://huggingface.co/InfoVis/InfoVis"><img src="https://img.shields.io/badge/InfoVis-🤗-yellow" alt="model"></a>
  <a href="https://huggingface.co/datasets/InfoVis/VisSlideBench"><img src="https://img.shields.io/badge/VisSlideBench-🤗-yellow" alt="benchmark"></a>
  <a href="https://huggingface.co/datasets/InfoVis/Data"><img src="https://img.shields.io/badge/InfoVis--Data-🤗-yellow" alt="data"></a>
</p>

<h4 align="center">
    <p>
        <a href=#-news>News</a> |
        <a href=#-quick-start>Quick Start</a> |
        <a href=#%EF%B8%8F-training>Training</a> |
        <a href=#%EF%B8%8F-citation>Citation</a>
    <p>
</h4>

<p align="center"><b>English</b> | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <img src="assets/overview.jpg" width="95%">
</p>

**InfoVis** is a unified 9B model for slide image generation and editing, fine-tuned from FLUX.2-klein-base-9B. One model handles text-to-slide generation, template-conditioned generation, multi-image grounded generation, and instruction-based slide editing.

## ✨ Highlights

- **One model, four slide tasks**: input images serve as a style reference, content assets, or the slide to edit.
- **Slide-native data and benchmark**: over 13M continual-pretraining and 1.9M SFT records, and **VisSlideBench**, 1,527 manually screened cases with task-specific metrics.
- **Strong results**: 7.94 overall on VisSlideBench against 7.99 for Nano-Banana-2, leading every evaluated open-source model on all eleven metrics.

## 🔥 News

- **2026-10**: Code, model weights, VisSlideBench with its evaluation code, and the SFT training data are released.

## 📖 Introduction

InfoVis is trained in three stages: slide-native continual pretraining, four-task mixed continual pretraining, and task-specific SFT whose branches are merged into one model. VisSlideBench scores generation on content faithfulness, instruction following, and aesthetic quality, and editing on semantic consistency and presentation quality.

| Model | Type | VisSlideBench Overall |
|---|---|---:|
| GPT-Image-2 | Proprietary | 8.36 |
| Nano-Banana-2 | Proprietary | 7.99 |
| FLUX.2-dev (32B) | Open-source | 6.37 |
| FLUX.2-klein-base-9B | Open-source | 6.30 |
| **InfoVis (9B)** | Open-source | **7.94** |

Per-task results and more baselines are in the paper.

## 📌 TODO

- [x] Training and inference code.
- [x] InfoVis model weights.
- [x] VisSlideBench and evaluation code.
- [x] SFT training data.
- [ ] Continual-pretraining data.

## 🚀 Quick Start

### 🛠️ Environment Setup

```bash
conda create -n infovis python=3.12
conda activate infovis

# Choose the command that matches your CUDA version (CUDA 12.6 shown here)
pip install torch==2.7.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/VectorSpaceLab/InfoVis.git
cd InfoVis
pip install -r requirements.txt
```

<details>
<summary>🌏 For users in Mainland China</summary>

```bash
pip install torch==2.7.1 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

### 🧪 Inference

Download the weights (or pass `--transformer InfoVis/InfoVis` to load them directly from Hugging Face):

```bash
hf download InfoVis/InfoVis --local-dir checkpoints/InfoVis
```

```bash
# Text-to-slide
python inference.py --transformer checkpoints/InfoVis \
  --prompt "A clean title slide for a talk on renewable energy, with a wind turbine illustration." \
  --out slide.png

# Slide editing
python inference.py --transformer checkpoints/InfoVis \
  --prompt "Change the title color to red." \
  --images source.png --width 1280 --height 720 --out edited.png
```

`--images` takes the reference slide (template-conditioned), one to four assets (multi-image grounded), or the source slide (editing). The base model `black-forest-labs/FLUX.2-klein-base-9B` is downloaded from Hugging Face on first use.

To run all four VisSlideBench tracks on 8 GPUs:

```bash
hf download InfoVis/VisSlideBench --repo-type dataset --local-dir VisSlideBench
bash scripts/infer_visslidebench.sh checkpoints/InfoVis VisSlideBench outputs/infovis
```

The scoring scripts and judge prompts are in `VisSlideBench/eval`; see its README.

## 🏋️ Training

One script runs every stage; each stage is a config in `configs/train/`:

```bash
NNODES=10 NODE_RANK=$RANK MASTER_ADDR=$MASTER bash scripts/train.sh configs/train/02_cpt_mix_a.yaml
```

The SFT data is available at [InfoVis/Data](https://huggingface.co/datasets/InfoVis/Data); the continual-pretraining data will follow. See [docs/training.md](docs/training.md) for the data format, all stage configs, and how to merge the SFT branches.

## 📜 License

The code is released under the [Apache-2.0 License](LICENSE). The InfoVis weights are a derivative of FLUX.2-klein-base-9B and follow the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B). VisSlideBench and the training data are released under CC BY-NC 4.0; see their dataset cards for details.

## 🙏 Acknowledgements

InfoVis builds on [FLUX.2-klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), [diffusers](https://github.com/huggingface/diffusers), and [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2).

## ❤️ Citation

If you find this repository or our work useful, please consider giving a star ⭐ and citation 🦖:

```bibtex
@article{wang2026infovis,
  title={InfoVis: Unified Slide Image Generation and Editing},
  author={Jiahao Wang and Xiyan Jiang and Chenyuan Wu and Haoqiang Zhang and Xin Luo and Yueze Wang and Defu Lian and Bo Zhang and Jiajun Zhang and Zheng Liu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
