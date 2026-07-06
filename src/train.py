"""
Single training entrypoint for both models — which one runs is decided by
`model:` in the yaml config, so the training loop, data loading and logging
are shared and only differ in which nn.Module + forward signature we call.

Usage:
    python src/train.py --config configs/vanilla.yaml
    python src/train.py --config configs/focus.yaml
"""
import argparse
import time

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import CelebAQueryDataset, collate
from losses import elbo_loss, estimate_bpp
from models.vanilla_vae import VanillaVAE
from models.focus_vae import FocusVAE
from utils import set_seed, get_device, save_checkpoint


def build_model(cfg, device):
    if cfg["model"] == "vanilla":
        return VanillaVAE(base_ch=cfg["base_ch"], feat_ch=cfg["feat_ch"],
                           z_ch=cfg["z_ch"], beta=cfg["beta"])
    elif cfg["model"] == "focus":
        return FocusVAE(base_ch=cfg["base_ch"], feat_ch=cfg["feat_ch"],
                         z_ch=cfg["z_ch"], beta_min=cfg["beta_min"],
                         beta_max=cfg["beta_max"], clip_name=cfg["clip_name"],
                         device=device)
    raise ValueError(f"Unknown model type: {cfg['model']}")


def step(model, batch, device, is_focus):
    images = batch["image"].to(device)
    if is_focus:
        out = model(images, queries=batch["query"])
    else:
        out = model(images)
    loss, parts = elbo_loss(out["recon"], images, out["mu"], out["logvar"], out["beta_map"])
    bpp = estimate_bpp(out["mu"], out["logvar"], images.shape[-2:]).mean().item()
    parts["bpp"] = bpp
    return loss, parts


@torch.no_grad()
def evaluate_val(model, loader, device, is_focus):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        loss, _ = step(model, batch, device, is_focus)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    is_focus = cfg["model"] == "focus"
    model = build_model(cfg, device).to(device)

    train_ds = CelebAQueryDataset(cfg["data_root"], cfg["image_size"], split="train")
    val_ds = CelebAQueryDataset(cfg["data_root"], cfg["image_size"], split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg["lr"])

    best_val = float("inf")
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            loss, parts = step(model, batch, device, is_focus)
            loss.backward()
            opt.step()
            if i % cfg["log_every"] == 0:
                print(f"epoch {epoch} step {i} "
                      f"total={parts['total']:.4f} recon={parts['recon']:.4f} "
                      f"kl={parts['kl']:.4f} bpp={parts['bpp']:.3f}")

        val_loss = evaluate_val(model, val_loader, device, is_focus)
        dt = time.time() - t0
        print(f"epoch {epoch} done in {dt:.1f}s val_loss={val_loss:.4f}")

        save_checkpoint(model, f"{cfg['ckpt_dir']}/last.pt", epoch=epoch, cfg=cfg)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, f"{cfg['ckpt_dir']}/best.pt", epoch=epoch, cfg=cfg)
            print(f"  new best (val_loss={val_loss:.4f}), saved checkpoint")


if __name__ == "__main__":
    main()
