"""
Loss functions shared by both models.

Rate-distortion framing:
    distortion  = reconstruction error (MSE)
    rate        = per-location KL(q(z|x) || p(z)), weighted by a beta map

For a uniform beta map this reduces to standard beta-VAE. For a spatially
varying beta map (FocusVAE) this is a query-conditioned rate allocation:
low beta where the query says "look here" (less penalty -> more bits kept),
high beta elsewhere (more penalty -> aggressive compression).
"""
import torch
import torch.nn.functional as F

LOG2 = 0.6931471805599453  # ln(2), for converting nats -> bits


def reconstruction_loss(recon, target):
    return F.mse_loss(recon, target, reduction="mean")


def kl_per_location(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, 1)) per spatial location, per channel-summed.
    Returns tensor of shape [B, H, W] (channels summed, nats)."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, z_ch, H, W]
    return kl.sum(dim=1)  # sum over channels -> [B, H, W]


def weighted_kl(mu, logvar, beta_map):
    """beta_map: [B, z_ch, H, W] or broadcastable to it. We average beta over
    the channel dim so the "how much do we penalize this location" scalar
    is per-(batch, H, W), matching kl_per_location's shape."""
    kl_map = kl_per_location(mu, logvar)               # [B, H, W]
    beta_scalar_map = beta_map.mean(dim=1)              # [B, H, W]
    return (beta_scalar_map * kl_map).sum(dim=(1, 2)).mean()  # scalar, mean over batch


def estimate_bpp(mu, logvar, image_hw):
    """Rate estimate in bits-per-pixel: total KL (nats) / ln(2) / num_pixels.
    This is the standard rate proxy used in learned image compression
    (Balle et al.) — under an idealized entropy coder the expected code
    length equals the KL divergence to the prior."""
    kl_map = kl_per_location(mu, logvar)               # [B, H, W]
    total_nats = kl_map.sum(dim=(1, 2))                 # [B]
    bits = total_nats / LOG2
    h, w = image_hw
    return bits / (h * w)                               # [B], bits per pixel


def elbo_loss(recon, target, mu, logvar, beta_map):
    recon_l = reconstruction_loss(recon, target)
    kl_l = weighted_kl(mu, logvar, beta_map)
    total = recon_l + kl_l
    return total, {"recon": recon_l.item(), "kl": kl_l.item(), "total": total.item()}
