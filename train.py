"""Full-parameter training of the FLUX.2-klein transformer for InfoVis.

The same script runs every stage (continued pre-training and the four SFT
branches); only the YAML config changes. Launch with accelerate + FSDP:

    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        --num_machines $NNODES --num_processes $((NNODES * 8)) \
        --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port 29500 \
        train.py --config configs/train/05_sft_t2i.yaml

Each checkpoint-N/ folder contains `pytorch_model_fsdp.bin` (full transformer
state dict), which the next stage loads through `init_from`.
"""

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import Flux2KleinPipeline
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from infovis.data import BucketBatchSampler, Collator, SlideDataset
from infovis.flow_matching import flow_matching_targets
from infovis.modeling import encode_text, load_frozen_encoders, load_transformer, pad_to, vae_encode

P = Flux2KleinPipeline  # static latent / position-id helpers


def lr_at(step, cfg):
    """Linear warmup from warmup_lr_init to learning_rate, then constant."""
    if step < cfg.warmup_steps:
        return cfg.warmup_lr_init + step * (cfg.learning_rate - cfg.warmup_lr_init) / cfg.warmup_steps
    return cfg.learning_rate


def build_inputs(batch, vae, dtype, device):
    """Encode one batch into the packed token sequence the transformer sees.

    Output latents are zero-padded to the largest size in the batch. Reference
    latents are padded to the same canvas and appended after the output tokens
    with time coordinate 10, 20, ... (FLUX.2 convention). Padded positions are
    removed from attention through the key mask.
    """
    out = [vae_encode(vae, img.to(device)).to(dtype) for img in batch["output_image"]]
    H, W = max(z.shape[1] for z in out), max(z.shape[2] for z in out)
    padded = [pad_to(z, H, W) for z in out]
    x_data = torch.stack([z for z, _ in padded])
    out_mask = torch.stack([m for _, m in padded]).to(device)

    t, x_t, target = flow_matching_targets(x_data)
    tokens, ids, masks = [P._pack_latents(x_t)], [P._prepare_latent_ids(x_data).to(device)], [out_mask]

    n_refs = max(len(r) if r else 0 for r in batch["input_images"])
    if n_refs:
        ref_tokens, ref_ids, ref_masks = [], [], []
        for refs in batch["input_images"]:
            refs = refs or []
            lats, ms = [], []
            for k in range(n_refs):
                if k < len(refs):
                    z, m = pad_to(vae_encode(vae, refs[k].to(device)).to(dtype), H, W)
                else:  # placeholder slot, fully masked
                    z, m = torch.zeros_like(x_data[0]), torch.zeros(H * W, dtype=torch.bool)
                lats.append(z)
                ms.append(m.to(device))
            ref_tokens.append(torch.cat([P._pack_latents(z[None])[0] for z in lats]))
            ref_ids.append(P._prepare_image_ids([z[None] for z in lats])[0])
            ref_masks.append(torch.cat(ms))
        tokens.append(torch.stack(ref_tokens))
        ids.append(torch.stack(ref_ids).to(device))
        masks.append(torch.stack(ref_masks))

    return {
        "t": t, "target": target, "out_mask": out_mask.view(-1, 1, H, W),
        "hidden_states": torch.cat(tokens, 1), "img_ids": torch.cat(ids, 1),
        "image_mask": torch.cat(masks, 1), "latent_ids": ids[0], "num_out_tokens": H * W,
    }


def save_checkpoint(accelerator, cfg, step, epoch, consumed):
    path = os.path.join(cfg.output_dir, f"checkpoint-{step}")
    accelerator.wait_for_everyone()
    accelerator.save_state(path)
    if accelerator.is_main_process:
        with open(os.path.join(path, "data_progress.json"), "w") as f:
            json.dump({"epoch": epoch, "consumed_batches": consumed, "global_step": step}, f)
        if cfg.train.get("checkpoints_total_limit"):
            ckpts = sorted((d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")),
                           key=lambda d: int(d.split("-")[1]))
            for d in ckpts[: -cfg.train.checkpoints_total_limit]:
                shutil.rmtree(os.path.join(cfg.output_dir, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.max_steps=100")
    args = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    tc, dc = cfg.train, cfg.data

    accelerator = Accelerator(gradient_accumulation_steps=tc.gradient_accumulation_steps,
                              mixed_precision="bf16", log_with="tensorboard", project_dir=cfg.output_dir)
    assert tc.batch_size * tc.gradient_accumulation_steps * accelerator.num_processes == tc.global_batch_size, \
        "batch_size * gradient_accumulation_steps * num_processes must equal global_batch_size"
    set_seed(cfg.seed, device_specific=True)
    device, dtype = accelerator.device, torch.bfloat16
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    tokenizer, text_encoder, vae = load_frozen_encoders(cfg.base_model, dtype, device)
    model = load_transformer(cfg.base_model, cfg.get("init_from"))
    model.train()
    if tc.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, betas=tuple(tc.adam_betas),
                                  weight_decay=tc.weight_decay, eps=tc.adam_eps)

    dataset = SlideDataset(dc.config, dc.root, max_output_pixels=dc.max_output_pixels,
                           max_input_pixels=dc.max_input_pixels, max_input_images=dc.max_input_images,
                           max_side_length=dc.max_side_length, prompt_dropout_prob=dc.prompt_dropout_prob,
                           ref_img_dropout_prob=dc.ref_img_dropout_prob)
    sampler = BucketBatchSampler(dataset, tc.batch_size, accelerator.num_processes, accelerator.process_index, cfg.seed)
    # The sampler already shards batches across ranks, so the loader is not passed to accelerator.prepare.
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=dc.num_workers,
                                         collate_fn=Collator(tokenizer, dc.max_tokens))
    model, optimizer = accelerator.prepare(model, optimizer)

    step, epoch, consumed = 0, 0, 0
    ckpts = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")] if os.path.isdir(cfg.output_dir) else []
    if ckpts:  # resume from the latest checkpoint of this run
        last = os.path.join(cfg.output_dir, max(ckpts, key=lambda d: int(d.split("-")[1])))
        accelerator.load_state(last)
        with open(os.path.join(last, "data_progress.json")) as f:
            prog = json.load(f)
        step, epoch, consumed = prog["global_step"], prog["epoch"], prog["consumed_batches"]
        accelerator.print(f"resumed from {last}")
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.name)

    offset = cfg.get("text_layer_offset", 1)
    bar = tqdm(total=tc.max_steps, initial=step, disable=not accelerator.is_local_main_process)
    while step < tc.max_steps:
        sampler.set_epoch(epoch, consumed)
        for batch in loader:
            with accelerator.accumulate(model):
                ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                prompt_embeds, txt_ids, txt_mask = encode_text(text_encoder, ids, mask, offset=offset, trim=True)
                x = build_inputs(batch, vae, dtype, device)
                attn_mask = torch.cat([txt_mask, x["image_mask"]], 1)[:, None, None, :]

                pred = model(hidden_states=x["hidden_states"], timestep=x["t"], guidance=None,
                             encoder_hidden_states=prompt_embeds, txt_ids=txt_ids, img_ids=x["img_ids"],
                             joint_attention_kwargs={"attention_mask": attn_mask}, return_dict=False)[0]
                pred = P._unpack_latents_with_ids(pred[:, : x["num_out_tokens"]], x["latent_ids"])

                # Squared error summed over valid (non-padded) latent elements, normalized by the
                # number of valid elements across all ranks; the factor undoes accelerate's division
                # by accumulation steps and FSDP's gradient averaging over ranks.
                valid = x["out_mask"].to(pred.dtype)
                sq = ((pred.float() - x["target"].float()) ** 2 * valid).sum()
                n_valid = accelerator.gather((valid.sum() * pred.shape[1]).float().reshape(1)).sum()
                loss = sq * tc.gradient_accumulation_steps * accelerator.num_processes / n_valid
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                    for g in optimizer.param_groups:
                        g["lr"] = lr_at(step, tc)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            consumed += accelerator.num_processes
            if accelerator.sync_gradients:
                step += 1
                bar.update(1)
                if step % tc.log_every == 0:
                    # mean over ranks / accumulation steps = per-element MSE of this micro-batch
                    logged = accelerator.gather(loss.detach().reshape(1)).mean().item() / tc.gradient_accumulation_steps
                    accelerator.log({"loss": logged, "lr": optimizer.param_groups[0]["lr"]}, step=step)
                if step % tc.checkpointing_steps == 0 or step == tc.max_steps:
                    save_checkpoint(accelerator, cfg, step, epoch, consumed)
                if step >= tc.max_steps:
                    break
        else:
            epoch, consumed = epoch + 1, 0
    accelerator.end_training()


if __name__ == "__main__":
    main()
