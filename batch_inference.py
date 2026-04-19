#!/usr/bin/env python3
"""Batch inference: run 50 seeds with a fixed prompt, reusing the loaded pipeline."""

import sys
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

SEEDS = [2852, 6190, 9089, 2750, 9747, 8999, 9541, 7718, 996, 85, 9657, 9977,
         8521, 8629, 9577, 5764, 898, 4421, 1622, 3002, 7648, 2050, 1472, 2101,
         5907, 8618, 7495, 715, 3886, 9531, 5093, 7350, 9225, 8307, 3092, 9690,
         9150, 7286, 6949, 9558, 8873, 2001, 3066, 6174, 753, 5339, 5252, 1739,
         6604, 6266, 4590, 8247, 8392, 2331, 3859, 1028, 108, 1930, 4568, 3718,
         3626, 3583, 268, 1391, 7053, 2214, 9925, 8858, 1325, 8174, 1765, 5110,
         6038, 3692, 5460, 6507, 4245, 9132, 7231, 6690, 6423, 5323, 6132, 9211,
         8039, 7358, 578, 8412, 1901, 1883, 240, 420, 1766, 1700, 7209, 4139,
         7207, 7958, 6087, 1182]

MODEL_SOURCE      = "./models/stable-diffusion-v1-5"
EMBEDDING_PATH    = "./textual_inversion_Karby_toy/learned_embeds-steps-500.safetensors"
PLACEHOLDER_TOKEN = "<Karby-toy>"
OUTPUT_DIR        = Path("./batch_seeds_output")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = (
    "(<Karby-toy>:1.4), MUST be only one cute pink bird-shaped toy with two small feet alone, "
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
print("Pipeline ready. Starting batch inference...\n")

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

print("\nAll done! Images saved to:", OUTPUT_DIR.resolve())
