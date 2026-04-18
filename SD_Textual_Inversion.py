#!/usr/bin/env python3
"""
SD Textual Inversion - Local execution script
Converted from Google Colab notebook:
  「SD_Textual_Inversion」的副本.ipynb

Usage:
  python SD_Textual_Inversion.py --mode all
  python SD_Textual_Inversion.py --mode train --max_train_steps 3000
  python SD_Textual_Inversion.py --mode inference --prompt "A photo of <cat-toy>"
"""

import os
import sys
import shutil
import subprocess
import glob
import argparse
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (mirror the Colab notebook defaults)
# ---------------------------------------------------------------------------
MODEL_NAME        = "runwayml/stable-diffusion-v1-5"
MODEL_LOCAL_DIR   = "./models/stable-diffusion-v1-5"
DATASET_REPO      = "diffusers/cat_toy_example"
DATA_DIR          = "./cat_toy_example"
OUTPUT_DIR        = "./textual_inversion_cat"
PLACEHOLDER_TOKEN = "<cat-toy>"
INITIALIZER_TOKEN = "toy"
TRAIN_SCRIPT_URL  = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main"
    "/examples/textual_inversion/textual_inversion.py"
)
TRAIN_SCRIPT_NAME = "textual_inversion.py"


# ---------------------------------------------------------------------------
# Step 1: GPU check
# ---------------------------------------------------------------------------
def check_gpu() -> None:
    print("\n=== GPU Info ===")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("[WARN] nvidia-smi not found. CUDA may be unavailable.")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[WARN] torch.cuda.is_available() = False. Training will be very slow on CPU.")
    except ImportError:
        print("[WARN] PyTorch not installed yet.")


# ---------------------------------------------------------------------------
# Step 2: Download the textual_inversion.py training script
# ---------------------------------------------------------------------------
def download_training_script() -> None:
    if Path(TRAIN_SCRIPT_NAME).exists():
        print(f"\n[SKIP] {TRAIN_SCRIPT_NAME} already exists.")
        return

    print(f"\n=== Downloading {TRAIN_SCRIPT_NAME} ===")
    response = requests.get(TRAIN_SCRIPT_URL, timeout=60)
    response.raise_for_status()
    Path(TRAIN_SCRIPT_NAME).write_text(response.text, encoding="utf-8")
    print(f"Saved to {TRAIN_SCRIPT_NAME}")


# ---------------------------------------------------------------------------
# Step 3: Download dataset and preview
# ---------------------------------------------------------------------------
def download_dataset() -> None:
    data_path = Path(DATA_DIR)

    if data_path.exists() and list(data_path.glob("*.jpeg")):
        print(f"\n[SKIP] Dataset already exists at {DATA_DIR}.")
        return

    print(f"\n=== Downloading dataset: {DATASET_REPO} ===")
    from huggingface_hub import snapshot_download

    snapshot_download(
        DATASET_REPO,
        local_dir=DATA_DIR,
        repo_type="dataset",
        ignore_patterns=".gitattributes",
    )

    # Remove hidden folders (e.g. .huggingface) to prevent training errors
    for hidden in data_path.glob(".*"):
        if hidden.is_dir():
            shutil.rmtree(hidden)
            print(f"Removed hidden folder: {hidden}")
        elif hidden.is_file():
            hidden.unlink()
            print(f"Removed hidden file: {hidden}")

    print(f"Dataset saved to {DATA_DIR}")


def preview_dataset() -> None:
    from PIL import Image

    images_paths = sorted(glob.glob(f"{DATA_DIR}/*.jpeg"))
    if not images_paths:
        print("[WARN] No JPEG images found in dataset directory.")
        return

    print(f"\n=== Dataset Preview: {len(images_paths)} images ===")
    for p in images_paths:
        img = Image.open(p)
        print(f"  {p}  {img.size}")


# ---------------------------------------------------------------------------
# Step 4: Download base model to local directory
# ---------------------------------------------------------------------------
def download_model() -> None:
    model_path = Path(MODEL_LOCAL_DIR)

    # Check if key model files already exist
    if model_path.exists() and any(model_path.glob("*.safetensors")):
        print(f"\n[SKIP] Base model already exists at {MODEL_LOCAL_DIR}.")
        return

    print(f"\n=== Downloading base model: {MODEL_NAME} (~4 GB) ===")
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_LOCAL_DIR,
        repo_type="model",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "rust_model*", "tf_model*"],
    )
    print(f"Model saved to {MODEL_LOCAL_DIR}")


# ---------------------------------------------------------------------------
# Step 5: Configure accelerate (one-time, non-interactive default config)
# ---------------------------------------------------------------------------
def configure_accelerate() -> None:
    print("\n=== Configuring accelerate ===")
    subprocess.run(["accelerate", "config", "default"], check=True)


# ---------------------------------------------------------------------------
# Step 6: Optional HuggingFace login (needed only for --push_to_hub)
# ---------------------------------------------------------------------------
def huggingface_login() -> None:
    print("\n=== HuggingFace Login ===")
    print("You need a write token from https://huggingface.co/settings/tokens")
    subprocess.run(["huggingface-cli", "login"], check=True)


# ---------------------------------------------------------------------------
# Step 7: Training
# ---------------------------------------------------------------------------
def train(args) -> None:
    if not Path(TRAIN_SCRIPT_NAME).exists():
        download_training_script()

    print(f"\n=== Training (max_train_steps={args.max_train_steps}) ===")

    # Use local model if downloaded, otherwise fall back to HuggingFace Hub
    model_source = MODEL_LOCAL_DIR if Path(MODEL_LOCAL_DIR).exists() else MODEL_NAME
    print(f"Model source: {model_source}")

    cmd = [
        "accelerate", "launch", TRAIN_SCRIPT_NAME,
        f"--pretrained_model_name_or_path={model_source}",
        f"--train_data_dir={DATA_DIR}",
        "--learnable_property=object",
        f"--placeholder_token={PLACEHOLDER_TOKEN}",
        f"--initializer_token={INITIALIZER_TOKEN}",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        f"--max_train_steps={args.max_train_steps}",
        "--learning_rate=5.0e-04",
        "--scale_lr",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--output_dir={OUTPUT_DIR}",
    ]

    if args.push_to_hub:
        cmd.append("--push_to_hub")

    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nTraining complete. Embedding saved to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Step 8: Inference
# ---------------------------------------------------------------------------
def inference(args) -> None:
    import torch
    from diffusers import StableDiffusionPipeline

    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        sys.exit(
            f"[ERROR] Output directory '{OUTPUT_DIR}' not found. "
            "Run training first with --mode train."
        )

    # Use local model if downloaded, otherwise fall back to HuggingFace Hub
    model_source = MODEL_LOCAL_DIR if Path(MODEL_LOCAL_DIR).exists() else MODEL_NAME

    print(f"\n=== Inference ===")
    print(f"Model    : {model_source}")
    print(f"Embedding: {OUTPUT_DIR}")
    print(f"Prompt   : {args.prompt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"\nLoading pipeline on {device} ({dtype}) ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)

    pipe.load_textual_inversion(OUTPUT_DIR)

    print("Generating image ...")
    image = pipe(prompt=args.prompt, num_inference_steps=args.num_inference_steps).images[0]

    out_path = Path(args.output_image)
    image.save(out_path)
    print(f"Image saved to: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Textual Inversion – local runner"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "setup", "train", "inference"],
        default="all",
        help=(
            "all    = setup + train + inference  (default)\n"
            "setup  = download script & dataset only\n"
            "train  = run training only\n"
            "inference = generate images only"
        ),
    )

    # Training options
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Training steps. Minimum 300 for usable results. (default: 3000)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained embedding to HuggingFace Hub (requires --login first)",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run huggingface-cli login before training",
    )

    # Inference options
    parser.add_argument(
        "--prompt",
        default=f"A photo of {PLACEHOLDER_TOKEN}",
        help=f"Text prompt for inference (default: 'A photo of {PLACEHOLDER_TOKEN}')",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps for inference (default: 50)",
    )
    parser.add_argument(
        "--output_image",
        default="output.png",
        help="File path to save the generated image (default: output.png)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    check_gpu()

    if args.mode in ("all", "setup"):
        download_training_script()
        download_dataset()
        preview_dataset()
        download_model()
        configure_accelerate()

    if args.login or (args.mode in ("all", "train") and args.push_to_hub):
        huggingface_login()

    if args.mode in ("all", "train"):
        train(args)

    if args.mode in ("all", "inference"):
        inference(args)


if __name__ == "__main__":
    main()
