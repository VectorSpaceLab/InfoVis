<h1 align="center">InfoVis: Unified Slide Image Generation and Editing</h1>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv%20paper-coming%20soon-b31b1b.svg" alt="arxiv">
  <a href="https://huggingface.co/InfoVis/InfoVis"><img src="https://img.shields.io/badge/InfoVis-🤗-yellow" alt="model"></a>
  <a href="https://huggingface.co/datasets/InfoVis/VisSlideBench"><img src="https://img.shields.io/badge/VisSlideBench-🤗-yellow" alt="benchmark"></a>
  <a href="https://huggingface.co/datasets/InfoVis/Data"><img src="https://img.shields.io/badge/InfoVis--Data-🤗-yellow" alt="data"></a>
</p>

<h4 align="center">
    <p>
        <a href=#-最新动态>最新动态</a> |
        <a href=#-快速开始>快速开始</a> |
        <a href=#%EF%B8%8F-训练>训练</a> |
        <a href=#%EF%B8%8F-引用>引用</a>
    <p>
</h4>

<p align="center"><a href="README.md">English</a> | <b>简体中文</b></p>

<p align="center">
  <img src="assets/overview.jpg" width="95%">
</p>

**InfoVis** 是一个统一的 9B 幻灯片图像生成与编辑模型，由 FLUX.2-klein-base-9B 微调得到。一个模型同时支持文生幻灯片、模板参考生成、多图约束生成和指令编辑四类任务。

## ✨ 亮点

- **一个模型覆盖四类幻灯片任务**：输入图像可以是风格参考页、需要放进页面的素材，或待编辑的原始页面。
- **面向幻灯片的数据与评测**：1300 万+ 条继续预训练数据和 190 万条 SFT 数据；**VisSlideBench** 包含 1,527 条人工筛选的测试样本，按任务设计评测指标。
- **效果领先**：VisSlideBench 总分 7.94（Nano-Banana-2 为 7.99），11 项指标全部领先所有参与评测的开源模型。

## 🔥 最新动态

- **2026-10**：发布代码、模型权重、VisSlideBench 及评测代码，以及 SFT 训练数据。

## 📖 简介

InfoVis 的训练分三个阶段：幻灯片继续预训练、四任务混合继续预训练，以及分任务 SFT 后把各分支融合成一个模型。VisSlideBench 对生成任务评测内容忠实度、指令遵循和美观度，对编辑任务评测语义一致性和呈现质量。

| 模型 | 类型 | VisSlideBench 总分 |
|---|---|---:|
| GPT-Image-2 | 闭源 | 8.36 |
| Nano-Banana-2 | 闭源 | 7.99 |
| FLUX.2-dev（32B） | 开源 | 6.37 |
| FLUX.2-klein-base-9B | 开源 | 6.30 |
| **InfoVis（9B）** | 开源 | **7.94** |

分任务结果和更多对比模型见论文。

## 📌 TODO

- [x] 训练代码与推理代码。
- [x] InfoVis 模型权重。
- [x] VisSlideBench 及评测代码。
- [x] SFT 训练数据。
- [ ] 继续预训练数据。

## 🚀 快速开始

### 🛠️ 环境配置

```bash
conda create -n infovis python=3.12
conda activate infovis

# 按 CUDA 版本选择对应命令（此处以 CUDA 12.6 为例）
pip install torch==2.7.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/VectorSpaceLab/InfoVis.git
cd InfoVis
pip install -r requirements.txt
```

<details>
<summary>🌏 中国大陆用户</summary>

```bash
pip install torch==2.7.1 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

### 🧪 推理

下载权重（也可以直接传 `--transformer InfoVis/InfoVis`，从 Hugging Face 加载）：

```bash
hf download InfoVis/InfoVis --local-dir checkpoints/InfoVis
```

```bash
# 文生幻灯片
python inference.py --transformer checkpoints/InfoVis \
  --prompt "A clean title slide for a talk on renewable energy, with a wind turbine illustration." \
  --out slide.png

# 幻灯片编辑
python inference.py --transformer checkpoints/InfoVis \
  --prompt "Change the title color to red." \
  --images source.png --width 1280 --height 720 --out edited.png
```

`--images` 传入参考页（模板参考生成）、1–4 张素材（多图约束生成）或原始页面（编辑）。首次运行时会从 Hugging Face 下载底模 `black-forest-labs/FLUX.2-klein-base-9B`。

用 8 张 GPU 跑完 VisSlideBench 的四个子任务：

```bash
hf download InfoVis/VisSlideBench --repo-type dataset --local-dir VisSlideBench
bash scripts/infer_visslidebench.sh checkpoints/InfoVis VisSlideBench outputs/infovis
```

打分脚本和评测 prompt 在 `VisSlideBench/eval` 下，用法见其中的 README。

## 🏋️ 训练

所有阶段共用一个训练脚本，每个阶段对应 `configs/train/` 下的一个配置：

```bash
NNODES=10 NODE_RANK=$RANK MASTER_ADDR=$MASTER bash scripts/train.sh configs/train/02_cpt_mix_a.yaml
```

SFT 数据已发布在 [InfoVis/Data](https://huggingface.co/datasets/InfoVis/Data)，继续预训练数据将稍后发布。数据格式、各阶段配置和 SFT 分支的融合方法见 [docs/training.md](docs/training.md)（英文）。

## 📜 许可证

代码采用 [Apache-2.0 许可证](LICENSE)。InfoVis 权重是 FLUX.2-klein-base-9B 的衍生模型，遵循 [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)。VisSlideBench 和训练数据采用 CC BY-NC 4.0 许可，详见各自的数据集说明。

## 🙏 致谢

InfoVis 基于 [FLUX.2-klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)、[diffusers](https://github.com/huggingface/diffusers) 和 [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) 构建。

## ❤️ 引用

如果本仓库或我们的工作对你有帮助，欢迎点个 star ⭐ 并引用 🦖：

```bibtex
@article{wang2026infovis,
  title={InfoVis: Unified Slide Image Generation and Editing},
  author={Jiahao Wang and Xiyan Jiang and Chenyuan Wu and Haoqiang Zhang and Xin Luo and Yueze Wang and Defu Lian and Bo Zhang and Jiajun Zhang and Zheng Liu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
