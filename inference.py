"""InfoVis inference: one slide image from an instruction and up to four images.

Single case:
    python inference.py --transformer checkpoints/InfoVis --prompt "A title slide ..." \
        [--images ref.png asset1.png] --width 1376 --height 768 --out slide.png

VisSlideBench (writes <out_dir>/<task>/<id>.png; shard over GPUs with --shard/--num_shards):
    python inference.py --transformer checkpoints/InfoVis --bench_root VisSlideBench \
        --task multi_image --out_dir outputs/infovis

Sampling follows the paper: 50 Euler steps, CFG 4.0 with an empty negative
prompt, 4,096 text tokens padded to full length and masked in attention, BF16,
seed = 2233 + case index. Output size follows the evaluation runs of the
released model: 1280x720 for t2i, and the record's width/height for template,
multi_image and edit (floored to multiples of 16).
"""

import argparse
import json
import os

import numpy as np
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from PIL import Image, ImageOps

from infovis.modeling import encode_text, tokenize, vae_decode


def load_pipeline(base_model, transformer=None, device="cuda"):
    pipe = Flux2KleinPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    if transformer:
        pipe.register_modules(transformer=Flux2Transformer2DModel.from_pretrained(transformer, torch_dtype=torch.bfloat16))
    return pipe.to(device)


def prepare_condition_images(pipe, images):
    """Cap each image at 1024x1024 pixels and center-crop to multiples of 32."""
    out = []
    for img in images:
        pipe.image_processor.check_image_input(img)
        if img.width * img.height > 1024 * 1024:
            img = pipe.image_processor._resize_to_target_area(img, 1024 * 1024)
        m = pipe.vae_scale_factor * 2
        out.append(pipe.image_processor.preprocess(img, height=img.height // m * m, width=img.width // m * m,
                                                   resize_mode="crop"))
    return out


def _embed(pipe, prompt, max_tokens, offset):
    t = tokenize(pipe.tokenizer, [prompt], max_tokens, padding="max_length")
    return encode_text(pipe.text_encoder, t["input_ids"].to(pipe.device), t["attention_mask"].to(pipe.device),
                       offset=offset)


@torch.no_grad()
def generate(pipe, prompt, images=None, width=1376, height=768, steps=50, guidance=4.0, seed=2233,
             max_tokens=4096, text_layer_offset=0):
    device, tr = pipe.device, pipe.transformer
    embeds, txt_ids, txt_mask = _embed(pipe, prompt, max_tokens, text_layer_offset)
    if guidance > 1:
        neg_embeds, neg_ids, neg_mask = _embed(pipe, "", max_tokens, text_layer_offset)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents, latent_ids = pipe.prepare_latents(1, tr.config.in_channels // 4, height, width, embeds.dtype,
                                               device, generator)
    cond, cond_ids = None, None
    if images:
        cond, cond_ids = pipe.prepare_image_latents(prepare_condition_images(pipe, images), 1, generator,
                                                    device, pipe.vae.dtype)

    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=steps)
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    pipe.scheduler.set_begin_index(0)

    n_out = latents.shape[1]
    img_ids = latent_ids if cond is None else torch.cat([latent_ids, cond_ids], 1)

    def predict(x, e, ids_txt, mask_txt, t):
        image_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        attn = torch.cat([mask_txt, image_mask], 1)[:, None, None, :]
        return tr(hidden_states=x, timestep=t / 1000, guidance=None, encoder_hidden_states=e, txt_ids=ids_txt,
                  img_ids=img_ids, joint_attention_kwargs={"attention_mask": attn}, return_dict=False)[0][:, :n_out]

    for t in pipe.progress_bar(timesteps):
        x = latents if cond is None else torch.cat([latents, cond], 1)
        x, tt = x.to(tr.dtype), t.expand(1).to(latents.dtype)
        v = predict(x, embeds, txt_ids, txt_mask, tt)
        if guidance > 1:
            v_neg = predict(x, neg_embeds, neg_ids, neg_mask, tt)
            v = v_neg + guidance * (v - v_neg)
        latents = pipe.scheduler.step(v, t, latents, return_dict=False)[0]

    latents = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)
    return pipe.image_processor.postprocess(vae_decode(pipe.vae, latents), output_type="pil")[0]


def open_image(path):
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def run_bench(pipe, args):
    with open(os.path.join(args.bench_root, "data", f"{args.task}.jsonl")) as f:
        records = [json.loads(line) for line in f]
    records = records[args.shard::args.num_shards]
    out_dir = os.path.join(args.out_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)
    for r in records:
        path = os.path.join(out_dir, f"{r['id']}.png")
        if os.path.exists(path):
            continue
        inputs = [r["source_image"]] if args.task == "edit" else r["input_images"]
        size = (1280, 720) if args.task == "t2i" else (r["width"], r["height"])
        width, height = (args.width, args.height) if args.width else size
        img = generate(pipe, r["instruction"], [open_image(os.path.join(args.bench_root, p)) for p in inputs],
                       width=width, height=height, steps=args.steps, guidance=args.guidance,
                       seed=args.seed + r["index"], max_tokens=args.max_tokens,
                       text_layer_offset=args.text_layer_offset)
        img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="black-forest-labs/FLUX.2-klein-base-9B")
    ap.add_argument("--transformer", default=None, help="InfoVis transformer folder (diffusers format)")
    ap.add_argument("--prompt")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--out", default="slide.png")
    ap.add_argument("--bench_root")
    ap.add_argument("--task", choices=["t2i", "template", "multi_image", "edit"])
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--width", type=int, default=None, help="override output width")
    ap.add_argument("--height", type=int, default=None, help="override output height")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=2233)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text_layer_offset", type=int, default=0, choices=[0, 1],
                    help="0: Qwen3 hidden_states[9,18,27] (used for the paper's evaluation); "
                         "1: outputs of decoder layers 9/18/27 (used during training)")
    args = ap.parse_args()

    pipe = load_pipeline(args.base_model, args.transformer, args.device)
    if args.bench_root:
        run_bench(pipe, args)
    else:
        img = generate(pipe, args.prompt, [open_image(p) for p in args.images], width=args.width or 1376,
                       height=args.height or 768, steps=args.steps, guidance=args.guidance, seed=args.seed,
                       max_tokens=args.max_tokens, text_layer_offset=args.text_layer_offset)
        img.save(args.out)


if __name__ == "__main__":
    main()
