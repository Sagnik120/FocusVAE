"""
FocusVAE = same backbone as VanillaVAE, plus a query-conditioned relevance
map that turns into a *spatially varying* beta:

    relevance[h, w]  = sigmoid(tau * cosine_sim(query_embedding, feat[:, h, w]))
    beta[h, w]       = beta_max - (beta_max - beta_min) * relevance[h, w]

High relevance -> low beta -> the KL penalty at that location is small, so
the encoder is free to keep more information (more bits) there.
Low relevance  -> high beta -> heavy penalty, latents get pulled toward the
prior (fewer effective bits, more compression).

`tau` is a learnable temperature (same trick as CLIP's logit_scale) so the
model can sharpen or soften the relevance map as training progresses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvEncoder, ConvDecoder, reparameterize
from .clip_encoder import FrozenCLIPTextEncoder


class FocusVAE(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, feat_ch=128, z_ch=16,
                 beta_min=0.1, beta_max=8.0, clip_name=None, device="cpu"):
        super().__init__()
        self.encoder = ConvEncoder(in_ch, base_ch, feat_ch, z_ch)
        self.decoder = ConvDecoder(in_ch, base_ch, z_ch)
        kwargs = {"feat_ch": feat_ch, "device": device}
        if clip_name:
            kwargs["clip_name"] = clip_name
        self.query_encoder = FrozenCLIPTextEncoder(**kwargs)

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.log_tau = nn.Parameter(torch.tensor(2.0))  # tau = exp(log_tau)

    def relevance_map(self, feat, queries):
        """feat: [B, feat_ch, H, W] -> relevance: [B, H, W] in [0, 1]"""
        q_emb = self.query_encoder(queries)                     # [B, feat_ch]
        feat_n = F.normalize(feat, dim=1)                        # [B, feat_ch, H, W]
        q_n = F.normalize(q_emb, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, feat_ch, 1, 1]
        cos_sim = (feat_n * q_n).sum(dim=1)                       # [B, H, W]
        tau = self.log_tau.exp()
        return torch.sigmoid(tau * cos_sim)                      # [B, H, W]

    def forward(self, x, queries, **kwargs):
        feat, mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)

        relevance = self.relevance_map(feat, queries)             # [B, H, W]
        beta_map_2d = self.beta_max - (self.beta_max - self.beta_min) * relevance
        beta_map = beta_map_2d.unsqueeze(1).expand_as(mu)         # [B, z_ch, H, W]

        return {
            "recon": recon, "mu": mu, "logvar": logvar,
            "beta_map": beta_map, "relevance": relevance,
        }
