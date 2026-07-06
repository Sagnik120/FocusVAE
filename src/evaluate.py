"""
Compare the baseline VanillaVAE against FocusVAE on:
  - overall PSNR / SSIM / bpp (whole image)
  - region-restricted PSNR (only the crop the query's region box points to)

The headline result this project is trying to show: at a similar overall
bit budget, FocusVAE reconstructs the query-relevant region *better* than
the baseline, because it's spending its bits there instead of uniformly.

Usage:
    python src/evaluate.py --vanilla_ckpt checkpoints/vanilla/best.pt \
                            --focus_ckpt checkpoints/focus/best.pt \
                            --data data/celeba
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import CelebAQueryDataset, collate
from losses import estimate_bpp
from models.vanilla_vae import VanillaVAE
from models.focus_vae import FocusVAE
from utils import get_device, load_checkpoint


def build_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    if cfg["model"] == "vanilla":
        model = VanillaVAE(base_ch=cfg["base_ch"], feat_ch=cfg["feat_ch"],
                            z_ch=cfg["z_ch"], beta=cfg["beta"])
    else:
        model = FocusVAE(base_ch=cfg["base_ch"], feat_ch=cfg["feat_ch"],
                          z_ch=cfg["z_ch"], beta_min=cfg["beta_min"],
                          beta_max=cfg["beta_max"], clip_name=cfg["clip_name"],
                          device=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg


def region_crop(img, box, image_size):
    """box: normalized [x1,y1,x2,y2] in [0,1]. img: [3, H, W] numpy."""
    h, w = image_size, image_size
    x1, y1, x2, y2 = box
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[:, y1:y2, x1:x2]


@torch.no_grad()
def run_eval(model, loader, device, is_focus, image_size):
    overall_psnr, overall_ssim, region_psnr, bpps = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        if is_focus:
            out = model(images, queries=batch["query"])
        else:
            out = model(images)

        recon = out["recon"].cpu().numpy()
        target = images.cpu().numpy()
        bpp = estimate_bpp(out["mu"], out["logvar"], images.shape[-2:]).cpu().numpy()
        bpps.extend(bpp.tolist())

        for i in range(recon.shape[0]):
            r, t = recon[i], target[i]
            overall_psnr.append(psnr(t, r, data_range=1.0))
            overall_ssim.append(ssim(t.transpose(1, 2, 0), r.transpose(1, 2, 0),
                                      data_range=1.0, channel_axis=2))
            box = batch["box"][i].numpy()
            rc, tc = region_crop(r, box, image_size), region_crop(t, box, image_size)
            if rc is not None and rc.shape[1] > 1 and rc.shape[2] > 1:
                region_psnr.append(psnr(tc, rc, data_range=1.0))

    return {
        "psnr": float(np.mean(overall_psnr)),
        "ssim": float(np.mean(overall_ssim)),
        "bpp": float(np.mean(bpps)),
        "region_psnr": float(np.mean(region_psnr)) if region_psnr else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla_ckpt", type=str, required=True)
    ap.add_argument("--focus_ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/celeba")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", type=str, default="results/comparison.md")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    vanilla, vcfg = build_from_ckpt(args.vanilla_ckpt, device)
    focus, fcfg = build_from_ckpt(args.focus_ckpt, device)
    image_size = vcfg["image_size"]

    test_ds = CelebAQueryDataset(args.data, image_size, split="test")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print("Evaluating baseline (VanillaVAE)...")
    v_res = run_eval(vanilla, loader, device, is_focus=False, image_size=image_size)
    print("Evaluating FocusVAE...")
    f_res = run_eval(focus, loader, device, is_focus=True, image_size=image_size)

    table = (
        "| Model | Overall PSNR | Overall SSIM | bpp | Region-restricted PSNR |\n"
        "|---|---|---|---|---|\n"
        f"| VanillaVAE (baseline) | {v_res['psnr']:.2f} | {v_res['ssim']:.3f} | "
        f"{v_res['bpp']:.4f} | {v_res['region_psnr']:.2f} |\n"
        f"| FocusVAE | {f_res['psnr']:.2f} | {f_res['ssim']:.3f} | "
        f"{f_res['bpp']:.4f} | {f_res['region_psnr']:.2f} |\n"
    )
    print(table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# FocusVAE vs VanillaVAE — comparison\n\n" + table +
        "\nRegion-restricted PSNR is measured only inside the crop the "
        "query's landmark-derived region box points to (e.g. mouth for "
        "\"is the person smiling?\"). The claim FocusVAE is making: at a "
        "similar or lower overall bpp, region-restricted PSNR should be "
        "higher than the baseline's.\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
