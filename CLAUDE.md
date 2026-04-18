# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a TibaMe course example for **Stable Diffusion Textual Inversion** — a fine-tuning technique that teaches a pre-trained SD model a new concept by learning a new text embedding token. The code is a single Google Colab notebook (`「SD_Textual_Inversion」的副本.ipynb`) designed to run on Colab with GPU access.

## Running the Notebook

The notebook is intended to run on **Google Colab** (not locally). Open it via the Colab badge in Cell 0, or upload it manually.

Execution order and key steps:

1. **GPU check** — `!nvidia-smi`
2. **Install dependencies** — installs `diffusers`, `accelerate`, `transformers` from pip, and `diffusers` from the HuggingFace `main` branch
3. **Download training script** — fetches `textual_inversion.py` from HuggingFace diffusers repo via `wget`
4. **Download dataset** — downloads `diffusers/cat_toy_example` from HuggingFace Hub (6 JPEG images); after download, delete hidden folders with `!rm -r ./cat_toy_example/.*`
5. **Configure accelerate** — `!accelerate config default`
6. **Login to HuggingFace** (optional, only if pushing to Hub) — `!huggingface-cli login` (requires a **write** token)
7. **Train** — launches `textual_inversion.py` via `accelerate launch`; default training uses `runwayml/stable-diffusion-v1-5`, placeholder token `<cat-toy>`, outputs to `./textual_inversion_cat/`
8. **Inference** — loads the trained embedding and generates images using the `<cat-toy>` token

## Training Configuration

Key training arguments used in Cell 22:

| Argument | Value |
|---|---|
| `--pretrained_model_name_or_path` | `runwayml/stable-diffusion-v1-5` |
| `--train_data_dir` | `./cat_toy_example` |
| `--learnable_property` | `object` |
| `--placeholder_token` | `<cat-toy>` |
| `--initializer_token` | `toy` |
| `--resolution` | 512 |
| `--max_train_steps` | 3000 (default); minimum **300** for usable results |
| `--learning_rate` | `5.0e-04` with `--scale_lr` |
| `--output_dir` | `textual_inversion_cat` |

The `--push_to_hub` flag is commented out by default.

## Local Python Script (SD_Textual_Inversion.py)

A standalone local script converted from the notebook. Requires a venv and CUDA GPU.

### One-time environment setup

```bash
# Create venv (already done)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install diffusers accelerate transformers huggingface_hub Pillow ftfy tensorboard safetensors requests
```

### Running the script

```bash
# Activate venv first
venv\Scripts\activate

# Full pipeline (setup → download → train → inference)
python SD_Textual_Inversion.py --mode all --max_train_steps 3000

# Only download dataset + training script + configure accelerate
python SD_Textual_Inversion.py --mode setup

# Training only (e.g. resume or re-run)
python SD_Textual_Inversion.py --mode train --max_train_steps 3000

# Inference only (training must have completed first)
python SD_Textual_Inversion.py --mode inference --prompt "A photo of <cat-toy>"

# With custom output filename and inference steps
python SD_Textual_Inversion.py --mode inference --prompt "A watercolor of <cat-toy>" --num_inference_steps 50 --output_image result.png

# Push trained embedding to HuggingFace Hub (requires write token)
python SD_Textual_Inversion.py --mode train --push_to_hub --login
```

### Script flow (mirrors notebook cells)

| Step | Function | Description |
|---|---|---|
| 1 | `check_gpu()` | `nvidia-smi` + torch CUDA check |
| 2 | `download_training_script()` | Downloads `textual_inversion.py` from HuggingFace via HTTP |
| 3 | `download_dataset()` | Downloads `diffusers/cat_toy_example`; removes hidden folders |
| 4 | `preview_dataset()` | Prints image filenames and sizes |
| 5 | `download_model()` | Downloads SD v1.5 base model (~4 GB) to `./models/stable-diffusion-v1-5` |
| 6 | `configure_accelerate()` | `accelerate config default` |
| 7 | `huggingface_login()` | `huggingface-cli login` (optional) |
| 8 | `train()` | `accelerate launch textual_inversion.py ...` |
| 9 | `inference()` | Loads embedding, generates and saves PNG |

## Important Notes

- After downloading the dataset, hidden subfolders (`.huggingface`) must be removed before training or an error will occur.
- Minimum recommended training steps is **300**; the original script default is **3000**.
- Push to Hub requires a HuggingFace **write** token.
- The notebook targets **SD v1.5** (`runwayml/stable-diffusion-v1-5`); there is commented-out code for SDXL with LoRA as an alternative.
