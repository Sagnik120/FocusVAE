"""
Baseline model: the same fully-convolutional, spatial-latent VAE as FocusVAE,
but with a *uniform* beta over every spatial location — i.e. no query, no
relevance map, standard beta-VAE rate-distortion tradeoff applied equally
everywhere. This isolates the effect of query-adaptive rate allocation when
compared against FocusVAE later.
"""
import torch.nn as nn

from .backbone import ConvEncoder, ConvDecoder, reparameterize


class VanillaVAE(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, feat_ch=128, z_ch=16, beta=1.0):
        super().__init__()
        self.encoder = ConvEncoder(in_ch, base_ch, feat_ch, z_ch)
        self.decoder = ConvDecoder(in_ch, base_ch, z_ch)
        self.beta = beta

    def forward(self, x, **kwargs):
        _, mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        # uniform beta map: same shape as mu/logvar spatial grid, all ones
        beta_map = x.new_full(mu.shape, self.beta)
        return {"recon": recon, "mu": mu, "logvar": logvar, "beta_map": beta_map}
