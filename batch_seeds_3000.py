#!/usr/bin/env python3
"""Batch inference: 100 random seeds with embedding steps=3000, reusing the loaded pipeline."""

import random
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

random.seed(42)
SEEDS = random.sample(range(10000), 100)

MODEL_SOURCE      = "./models/stable-diffusion-v1-5"
EMBEDDING_PATH    = "./textual_inversion_Karby_toy/learned_embeds-steps-3000.safetensors"
PLACEHOLDER_TOKEN = "<Karby-toy>"
OUTPUT_DIR        = Path("./batch_seeds_output_3000")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = (
    "(<Karby-toy>:1.4), only one cute pink bird-shaped toy with two small feet alone, "
    "soaring through a bright blue sky above green mountains, surrounded by fluffy white clouds, "
    "wings spread wide, cartoon style, cheerful expression, vibrant colors, sunny daytime, "
    "aerial view, highly detailed, masterpiece, best quality"
)
NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, ugly, watermark, text, dark, indoor, table, floor, ground, "
    "stars, multiple characters, extra limbs, mutated, multiple toys, extra limbs, extra feet, "
    "deformed wings, distorted body, duplicate, blurry, low quality, bad anatomy, realistic bird, "
    "text, watermark, cropped, out of frame"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"Loading pipeline on {device} ({dtype}) ...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_SOURCE,
    torch_dtype=dtype,
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

pipe.load_textual_inversion(EMBEDDING_PATH, token=PLACEHOLDER_TOKEN)
print(f"Embedding loaded: {EMBEDDING_PATH}")
print(f"Pipeline ready. Running {len(SEEDS)} seeds...\n")

for i, seed in enumerate(SEEDS, 1):
    out_path = OUTPUT_DIR / f"seed_{seed:05d}.png"
    if out_path.exists():
        print(f"[{i:03d}/100] seed={seed} already exists, skipping.")
        continue
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=75,
        guidance_scale=13,
        generator=generator,
    ).images[0]
    image.save(out_path)
    print(f"[{i:03d}/100] seed={seed:5d} -> {out_path}", flush=True)

print(f"\nAll done! Images saved to: {OUTPUT_DIR.resolve()}")
print(f"Seeds used: {SEEDS}")
