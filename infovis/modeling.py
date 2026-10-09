"""Model components and latent / text helpers shared by training and inference.

InfoVis keeps the FLUX.2-klein-base-9B architecture unchanged, so every module
comes from diffusers / transformers:
  - Flux2Transformer2DModel (the only trained module)
  - AutoencoderKLFlux2 (frozen)
  - Qwen3 text encoder + Qwen2 tokenizer (frozen)
"""

import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

# Output resolutions FLUX.2-klein prefers; used as training buckets (height, width).
PREFERRED_KLEIN_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328), (832, 1248),
    (880, 1184), (944, 1104), (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]

TEXT_LAYERS = (9, 18, 27)  # indices into Qwen3 `hidden_states`, as in the FLUX.2-klein pipeline


def load_frozen_encoders(base_model, dtype, device):
    tokenizer = Qwen2TokenizerFast.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLFlux2.from_pretrained(base_model, subfolder="vae")
    for m in (text_encoder, vae):
        m.requires_grad_(False).eval().to(device, dtype=dtype)
    return tokenizer, text_encoder, vae


def load_transformer(base_model, init_from=None):
    """Load the trainable transformer.

    `init_from` is None (base weights), a diffusers transformer folder, or a
    full state dict saved by a previous stage (`checkpoint-N/pytorch_model_fsdp.bin`).
    """
    if init_from and os.path.isdir(init_from):
        return Flux2Transformer2DModel.from_pretrained(init_from)
    model = Flux2Transformer2DModel.from_pretrained(base_model, subfolder="transformer")
    if init_from:
        sd = torch.load(init_from, map_location="cpu", mmap=True, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print(f"loaded {init_from}: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def tokenize(tokenizer, prompts, max_length, padding):
    """Qwen chat template without thinking, as used for both training and inference."""
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=padding, truncation=True, max_length=max_length)


@torch.no_grad()
def encode_text(text_encoder, input_ids, attention_mask, layers=TEXT_LAYERS, offset=0, trim=False):
    """Stack three Qwen3 hidden states into (B, L, 3 * 4096) prompt embeddings.

    `offset=0` takes `hidden_states[k]` (the input of decoder layer k), which is
    what the FLUX.2-klein pipeline and our evaluation use. `offset=1` takes the
    output of decoder layer k (= `hidden_states[k + 1]`); all InfoVis training
    runs reported in the paper used `offset=1`.
    With `trim=True` the sequence is cut to the longest unpadded prompt and the
    padded positions are zeroed (training); inference keeps full padding and
    relies on the attention mask.
    """
    captured, hooks = {}, []
    for k in layers:
        def hook(_m, _i, out, k=k):
            captured[k] = out[0] if isinstance(out, tuple) else out

        hooks.append(text_encoder.model.layers[k - 1 + offset].register_forward_hook(hook))
    try:
        # Run the decoder only; the LM head is not needed and its logits are large for long prompts.
        text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    if trim:
        length = max(int(attention_mask.sum(1).max()), 1)
        attention_mask = attention_mask[:, :length]
    else:
        length = attention_mask.shape[1]
    embeds = torch.cat([captured[k][:, :length] for k in layers], dim=-1)
    if trim:
        embeds = embeds * attention_mask.unsqueeze(-1).to(embeds.dtype)
    text_ids = Flux2KleinPipeline._prepare_text_ids(embeds).to(embeds.device)
    return embeds, text_ids, attention_mask.bool()


@torch.no_grad()
def vae_encode(vae, image):
    """Pixels in [-1, 1] (1, 3, H, W) -> normalized, patchified latent (128, H/16, W/16)."""
    z = vae.encode(image.to(vae.dtype)).latent_dist.mode()
    z = Flux2KleinPipeline._patchify_latents(z)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(z) + vae.config.batch_norm_eps)
    return ((z - mean) / std).squeeze(0)


@torch.no_grad()
def vae_decode(vae, latents):
    """Inverse of `vae_encode` for a (B, 128, h, w) batch; returns pixels in [-1, 1]."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(latents) + vae.config.batch_norm_eps)
    latents = Flux2KleinPipeline._unpatchify_latents(latents * std + mean)
    return vae.decode(latents.to(vae.dtype), return_dict=False)[0]


def pad_to(latent, h, w):
    """Zero-pad a (C, h0, w0) latent to (C, h, w); returns it with a token mask of length h*w."""
    _, h0, w0 = latent.shape
    if h0 > h or w0 > w:  # larger than the target canvas: resize instead of padding
        return F.interpolate(latent[None], size=(h, w), mode="bilinear", align_corners=False)[0], torch.ones(h * w, dtype=torch.bool)
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[:h0, :w0] = True
    return F.pad(latent, (0, w - w0, 0, h - h0)), mask.flatten()
