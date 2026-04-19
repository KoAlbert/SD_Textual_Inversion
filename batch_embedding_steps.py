#!/usr/bin/env python3
"""
Batch inference over embedding steps 500-3000 (step 500).
Loads the pipeline once and swaps embeddings for each checkpoint.
"""

import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

EMBEDDING_STEPS    = [500, 1000, 1500, 2000, 2500, 3000]
MODEL_SOURCE       = "./models/stable-diffusion-v1-5"
OUTPUT_DIR_BASE    = "./textual_inversion_Karby_toy"
PLACEHOLDER_TOKEN  = "<Karby-toy>"
SEED               = 8618
NUM_INFERENCE_STEPS = 75
GUIDANCE_SCALE     = 13.0

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

print(f"Loading base pipeline on {device} ({dtype}) ...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_SOURCE,
    torch_dtype=dtype,
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

print("Pipeline ready. Starting batch over embedding steps...\n")

for steps in EMBEDDING_STEPS:
    embedding_path = Path(OUTPUT_DIR_BASE) / f"learned_embeds-steps-{steps}.safetensors"
    out_path = Path(f"karby_bird_sky_{SEED}_steps{steps}.png")

    if not embedding_path.exists():
        print(f"[SKIP] Embedding not found: {embedding_path}")
        continue

    # Reset tokenizer/text encoder to base state, then load this checkpoint's embedding
    pipe.tokenizer = pipe.tokenizer.__class__.from_pretrained(
        MODEL_SOURCE, subfolder="tokenizer"
    )
    pipe.text_encoder = pipe.text_encoder.__class__.from_pretrained(
        MODEL_SOURCE, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)

    pipe.load_textual_inversion(str(embedding_path), token=PLACEHOLDER_TOKEN)

    generator = torch.Generator(device=device).manual_seed(SEED)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    image.save(out_path)
    print(f"[steps={steps:4d}] saved -> {out_path.resolve()}", flush=True)

print("\nAll done!")
