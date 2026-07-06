"""
Shared fully-convolutional encoder/decoder used by both the baseline
(uniform-beta) VAE and FocusVAE. Keeping this identical between the two
models means any difference in results comes from the query-conditioned
rate allocation, not from architecture differences.

The latent is a *spatial grid*, not a flattened vector: for a 64x64 input
with three stride-2 downsamples we get an 8x8 grid of latent vectors, each
of dimension `z_ch`. This is what lets us allocate bits per-location instead
of per-image.
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def deconv_block(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvEncoder(nn.Module):
    """64x64x3 -> feature map [B, feat_ch, 8, 8] -> (mu, logvar) [B, z_ch, 8, 8]"""

    def __init__(self, in_ch=3, base_ch=32, feat_ch=128, z_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, base_ch, stride=2),      # 64 -> 32
            conv_block(base_ch, base_ch * 2, stride=2),  # 32 -> 16
            conv_block(base_ch * 2, feat_ch, stride=2),  # 16 -> 8
        )
        self.to_mu = nn.Conv2d(feat_ch, z_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(feat_ch, z_ch, kernel_size=1)

    def forward(self, x):
        feat = self.net(x)                 # [B, feat_ch, 8, 8]
        mu = self.to_mu(feat)              # [B, z_ch, 8, 8]
        logvar = self.to_logvar(feat)      # [B, z_ch, 8, 8]
        return feat, mu, logvar


class ConvDecoder(nn.Module):
    """[B, z_ch, 8, 8] -> 64x64x3 reconstruction (sigmoid output)"""

    def __init__(self, out_ch=3, base_ch=32, z_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            deconv_block(z_ch, base_ch * 2, stride=2),   # 8 -> 16
            deconv_block(base_ch * 2, base_ch, stride=2),  # 16 -> 32
            nn.ConvTranspose2d(base_ch, out_ch, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
