"""
Push a trained checkpoint to the Hugging Face Hub as a model repo.

Usage:
    huggingface-cli login   # once, with your Sagnik120 HF account token
    python scripts/push_to_hub.py --repo Sagnik120/focus-vae --ckpt checkpoints/focus/best.pt
"""
import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo

MODEL_CARD = """---
license: mit
tags:
  - vae
  - image-compression
  - clip
  - pytorch
---

# FocusVAE

Query-adaptive semantic image compression: a convolutional VAE whose
per-region bit allocation is conditioned on a natural-language query via a
frozen CLIP text encoder. See the training repo for full details:
https://github.com/Sagnik120/FocusVAE

## Usage
```python
import torch
ckpt = torch.load("focus_vae.pt", map_location="cpu")
cfg = ckpt["cfg"]
# rebuild the FocusVAE class from the training repo, then:
# model.load_state_dict(ckpt["model_state"])
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, required=True, help="e.g. Sagnik120/focus-vae")
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    staging = Path("hf_staging")
    staging.mkdir(exist_ok=True)
    shutil.copy(ckpt_path, staging / "focus_vae.pt")
    (staging / "README.md").write_text(MODEL_CARD)

    create_repo(args.repo, exist_ok=True)
    api = HfApi()
    api.upload_folder(folder_path=str(staging), repo_id=args.repo, repo_type="model")
    print(f"Pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
