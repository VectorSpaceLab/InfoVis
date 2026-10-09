"""Training data: jsonl sources -> resized image tensors, grouped into resolution buckets.

A data config (configs/data/*.yaml) lists jsonl sources:

    sources:
      - path: sft/t2i/short.jsonl     # relative to the data root
        ratio: 1.0                    # 1 = use all, <1 = subsample, >1 = oversample

Each jsonl line is one sample:

    {"id": "...", "task": "t2i" | "template" | "multi_image" | "edit",
     "instruction": "...",
     "input_images": ["images/....png", ...],   # [] for text-to-slide
     "output_image": "images/....png",
     "target_img_size": [width, height]}

Image paths are relative to the data root unless absolute. Files must contain
exactly these fields (a missing `id` or `task` is filled with null).
"""

import math
import os
import random
import re

import torch
import yaml
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.training_utils import find_nearest_bucket
from PIL import Image

from .modeling import PREFERRED_KLEIN_RESOLUTIONS, tokenize

# Fixed schema: otherwise a file whose `input_images` are all empty loads as a
# null-typed column and cannot be concatenated with the other sources.
FEATURES = Features({
    "id": Value("string"), "task": Value("string"), "instruction": Value("string"),
    "input_images": Sequence(Value("string")), "output_image": Value("string"),
    "target_img_size": Sequence(Value("int64")),
})

_PLACEHOLDERS = ["<img>", "</img>"] + [f"<|image_{i}|>" for i in range(1, 6)] + [f"<|img_{i}|>" for i in range(1, 6)]
_CAPTION_PREFIXES = ["The image portrays ", "The image depicts ", "The image captures ",
                     "The image highlights ", "The image shows "]


def normalize_whitespace(s):
    s = re.sub(r" +\t+", "\n", s)
    s = re.sub(r" +\n+", "\n", s)
    s = re.sub(r"\t+ +", "\n", s)
    s = re.sub(r"\n+ +", "\n", s)
    s = re.sub(r" +", " ", s)
    s = re.sub(r"\t+", "\t", s)
    s = re.sub(r"\n+", "\n", s)
    return s.strip()


def clean_instruction(text):
    if random.random() < 0.5:
        for p in _CAPTION_PREFIXES:
            if p in text:
                text = text.replace(p, "")
                break
    for w in _PLACEHOLDERS:
        text = text.replace(w, "")
    return normalize_whitespace(text)


def _resolve(path, root):
    return path if os.path.isabs(path) else os.path.join(root, path)


class SlideDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, data_root, max_output_pixels=1024 * 1024, max_input_pixels=(1024 * 1024,),
                 max_input_images=4, max_side_length=2048, prompt_dropout_prob=0.0, ref_img_dropout_prob=0.0,
                 num_proc=8):
        with open(data_config) as f:
            cfg = yaml.safe_load(f)
        self.root = data_root
        self.max_output_pixels = max_output_pixels
        self.max_input_pixels = list(max_input_pixels)
        self.max_input_images = max_input_images
        self.max_side_length = max_side_length
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        self.processor = Flux2ImageProcessor(vae_scale_factor=16, do_resize=True)

        parts, source_ids = [], []
        for i, src in enumerate(cfg["sources"]):
            ds = load_dataset("json", data_files=_resolve(src["path"], data_root), features=FEATURES,
                              num_proc=num_proc)["train"]
            ratio = float(src.get("ratio", 1.0))
            if ratio != 1.0:
                k = int(len(ds) * ratio)
                idx = random.sample(range(len(ds)), k) if k <= len(ds) else random.choices(range(len(ds)), k=k)
                ds = ds.select(idx)
            parts.append(ds)
            source_ids += [i] * len(ds)
            print(f"[data] {src['path']}: {len(ds)} samples")
        self.data = concatenate_datasets(parts)

        # Bucket by (output resolution, source) so a batch never mixes sources;
        # this keeps long style_json prompts from padding short ones.
        n_src = len(parts)
        self.bucket_indices = [[] for _ in range(len(PREFERRED_KLEIN_RESOLUTIONS) * n_src)]
        for idx, (size, src) in enumerate(zip(self.data["target_img_size"], source_ids)):
            w, h = size
            b = find_nearest_bucket(h, w, PREFERRED_KLEIN_RESOLUTIONS)
            self.bucket_indices[b * n_src + src].append(idx)

    def __len__(self):
        return len(self.data)

    def _fit(self, w, h, max_pixels):
        """Scale (w, h) down to fit max_side_length / max_pixels, floored to multiples of 16."""
        r = min(1.0, self.max_side_length / max(w, h))
        if max_pixels and w * h > max_pixels:
            r = min(r, math.sqrt(max_pixels / (w * h)))
        w, h = max(1, int(w * r)), max(1, int(h * r))
        return max(16, w // 16 * 16), max(16, h // 16 * 16)

    def _load(self, path, size, max_pixels):
        img = Image.open(_resolve(path, self.root)).convert("RGB")
        w, h = self._fit(*(size or img.size), max_pixels)
        return self.processor.preprocess(img, height=h, width=w, resize_mode="default")

    def _get(self, index):
        item = self.data[index]
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_refs = drop_prompt and random.random() < self.ref_img_dropout_prob
        instruction = "" if drop_prompt else clean_instruction(item["instruction"])

        inputs = None
        paths = (item["input_images"] or [])[: self.max_input_images]
        if paths and not drop_refs:
            max_pixels = self.max_input_pixels[min(len(paths), len(self.max_input_pixels)) - 1]
            inputs = []
            for p in paths:
                img = Image.open(_resolve(p, self.root)).convert("RGB")
                self.processor.check_image_input(img)
                w, h = self._fit(*img.size, max_pixels)
                inputs.append(self.processor.preprocess(img, height=h, width=w, resize_mode="default"))

        output = self._load(item["output_image"], item["target_img_size"], self.max_output_pixels)
        return {"instruction": instruction, "input_images": inputs, "output_image": output}

    def __getitem__(self, index):
        for attempt in range(100):  # skip unreadable samples
            try:
                return self._get(index)
            except Exception as e:
                print(f"[data] failed to load sample {index}: {e}")
                index = random.randint(0, len(self) - 1)
        raise RuntimeError("too many unreadable samples")


class Collator:
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, batch):
        text = tokenize(self.tokenizer, [b["instruction"] for b in batch], self.max_tokens, padding="longest")
        return {
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "input_images": [b["input_images"] for b in batch],
            "output_image": [b["output_image"] for b in batch],
        }


class BucketBatchSampler(torch.utils.data.Sampler):
    """Distributed batch sampler over dataset.bucket_indices.

    The global batch order depends only on (seed, epoch), so a run can resume
    on a different number of GPUs by skipping the batches already consumed.
    """

    def __init__(self, dataset, batch_size, num_replicas, rank, seed=0):
        self.dataset, self.batch_size = dataset, batch_size
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.epoch, self.consumed = 0, 0
        self.num_batches = len(self._batches(0))

    def _batches(self, epoch):
        g = torch.Generator().manual_seed(self.seed + epoch)
        batches = []
        for bucket in self.dataset.bucket_indices:
            order = [bucket[i] for i in torch.randperm(len(bucket), generator=g).tolist()]
            batches += [order[i:i + self.batch_size] for i in range(0, len(order) - self.batch_size + 1, self.batch_size)]
        return [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]

    def __iter__(self):
        yield from self._batches(self.epoch)[self.consumed:][self.rank::self.num_replicas]

    def __len__(self):
        return max(0, self.num_batches - self.consumed) // self.num_replicas

    def set_epoch(self, epoch, consumed=0):
        self.epoch, self.consumed = epoch, consumed
