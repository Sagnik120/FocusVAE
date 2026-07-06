# FocusVAE — Query-Adaptive Semantic Image Compression

FocusVAE is a fully-convolutional VAE whose **bit allocation is conditioned on a
natural-language query**. Instead of spending latent capacity uniformly over
every pixel (like a standard VAE / JPEG), FocusVAE uses a frozen CLIP text
encoder to figure out *which regions of the image the query cares about*, and
spends more bits there while compressing the rest of the image harder.

> "Is the person smiling?" → bits go to the mouth region.
> "Is the person wearing glasses?" → bits go to the eyes region.
> Same total bit budget, task-relevant reconstruction quality goes up.

This project sits next to my M.Tech thesis work on query-adaptive visual
processing for VLMs (FocusVLM — token pruning). FocusVAE explores the same
idea — *let the query decide where the model spends compute/bits* — but for
image compression/reconstruction instead of token pruning.

## Why this is useful in the real world
Bandwidth- and storage-constrained vision pipelines (edge cameras, satellite
imagery triage, IoT sensors) usually don't need every pixel reconstructed
perfectly — they need the parts relevant to a downstream task (is there a
person / is a valve open / is the driver looking at the road) reconstructed
well, and can tolerate heavy compression everywhere else. FocusVAE gives a
trainable, query-conditioned knob for exactly that tradeoff.

## How it works
1. A convolutional encoder maps the image to a **spatial** latent grid
   `(mu, logvar)` of shape `[C_z, H', W']` — not a single flattened vector.
2. A frozen CLIP text encoder embeds the query. A small trainable linear
   head projects it to the same channel dimension as the encoder's feature
   map.
3. Cosine similarity between the projected query embedding and each spatial
   location of the feature map gives a **relevance map** `R ∈ [0,1]^{H'×W'}`.
4. The KL term of the ELBO is weighted **per spatial location** by a
   relevance-dependent β: low β (small penalty → more information kept) where
   `R` is high, high β (heavy penalty → aggressively compressed) where `R` is
   low.
5. Rate is estimated the standard way in learned-compression literature: the
   expected KL divergence approximates the entropy-coding cost of the latents
   (bits-per-pixel).

The **baseline** model is the exact same architecture with a *uniform* β
(no query, no relevance map) — this isolates the effect of query-adaptive
rate allocation instead of confounding it with architecture differences.

## Dataset
**CelebA** (aligned & cropped, 64×64/128×128). Chosen over COCO for this
lean scope because:
- It ships with 40 binary attributes → free, natural-language queries
  ("does the person have a mustache?", "is the person smiling?") via simple
  templates, no captioning pipeline needed.
- It ships with 5-point facial landmarks → lets us evaluate reconstruction
  quality *specifically in the query-relevant region* (e.g. crop around the
  mouth for a "smiling" query) without any extra annotation work.
- Small image size + small subset (5k–20k images) trains comfortably on an
  Apple Silicon MacBook (M5, 24GB unified memory, MPS backend) in a lean
  timeframe.

## Repo layout
```
FocusVAE/
├── data/
│   ├── download_celeba.py     # pulls CelebA via Kaggle CLI, subsets it
│   └── prepare_queries.py     # attribute -> natural language query templates
├── src/
│   ├── dataset.py              # CelebA Dataset + landmark-based region crops
│   ├── losses.py                # recon loss, per-location KL, spatial beta
│   ├── utils.py                 # seeding, checkpointing, bpp estimation
│   ├── models/
│   │   ├── backbone.py          # shared conv encoder/decoder (spatial latent)
│   │   ├── vanilla_vae.py        # baseline: uniform-beta spatial VAE
│   │   ├── clip_encoder.py       # frozen CLIP text encoder + projection head
│   │   └── focus_vae.py          # query-conditioned relevance map + spatial beta
│   ├── train.py                  # single entrypoint, --model {vanilla,focus}
│   └── evaluate.py               # PSNR/SSIM/bpp, overall vs region-restricted
├── configs/
│   ├── vanilla.yaml
│   └── focus.yaml
├── demo/
│   └── app.py                    # Gradio demo: image + query -> heatmap + recon
├── scripts/
│   ├── run_baseline.sh
│   └── run_focus.sh
└── requirements.txt
```

## Setup (macOS, Apple Silicon)
```bash
git clone https://github.com/Sagnik120/FocusVAE.git
cd FocusVAE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get the data
```bash
# needs ~/.kaggle/kaggle.json with your Kaggle API token (account: sagnikchandra027)
python data/download_celeba.py --out data/celeba --subset 15000
python data/prepare_queries.py --root data/celeba
```

## 2. Train the baseline (uniform-beta spatial VAE)
```bash
python src/train.py --config configs/vanilla.yaml
```

## 3. Train FocusVAE (query-adaptive beta)
```bash
python src/train.py --config configs/focus.yaml
```

Both scripts auto-detect and use the MPS backend (`torch.device("mps")`) if
available, falling back to CPU otherwise.

## 4. Evaluate
```bash
python src/evaluate.py --vanilla_ckpt checkpoints/vanilla/best.pt \
                        --focus_ckpt checkpoints/focus/best.pt \
                        --data data/celeba
```
This prints overall PSNR/SSIM/bpp for both models plus **region-restricted**
PSNR (using landmark crops) for each query type — the headline comparison
that shows FocusVAE wins where it matters (relevant region) while trading
off background quality it doesn't need.

## 5. Run the demo
```bash
python demo/app.py
```
Upload an image, type a query, see the relevance heatmap and both
reconstructions side by side.

## 6. Push the trained model to Hugging Face
```bash
pip install huggingface_hub
huggingface-cli login   # HF account: Sagnik120
python scripts/push_to_hub.py --repo Sagnik120/focus-vae --ckpt checkpoints/focus/best.pt
```

## Results
_Fill in after training on your machine — `evaluate.py` writes a
`results/comparison.md` table you can paste here._

## Status / roadmap
- [x] Data pipeline + query templates
- [x] Baseline spatial VAE
- [x] CLIP-conditioned relevance map + spatial-beta FocusVAE
- [x] Evaluation (overall + region-restricted)
- [x] Gradio demo
- [ ] Push weights to Hugging Face Hub
- [ ] (stretch) Try COCO captions instead of CelebA attributes for open-vocabulary queries

## License
MIT — see `LICENSE`.
