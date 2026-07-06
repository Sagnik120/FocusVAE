"""
Frozen CLIP text encoder + a small trainable projection head that maps the
CLIP embedding into the same channel dimension as the image encoder's
feature map, so we can take a cosine similarity against every spatial
location.

CLIP itself is never fine-tuned here (frozen, eval mode, no grad) — only
`proj` is trained. This keeps the extra parameter count and compute tiny,
which matters on a 24GB Apple Silicon machine.
"""
import torch
import torch.nn as nn
from transformers import CLIPTokenizerFast, CLIPTextModelWithProjection

DEFAULT_CLIP = "openai/clip-vit-base-patch32"


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, feat_ch=128, clip_name=DEFAULT_CLIP, device="cpu"):
        super().__init__()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(clip_name)
        self.clip = CLIPTextModelWithProjection.from_pretrained(clip_name)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        clip_dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, feat_ch),
            nn.LayerNorm(feat_ch),
        )
        self.device_ = device

    @torch.no_grad()
    def _clip_embed(self, queries):
        tokens = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(self.device_) for k, v in tokens.items()}
        out = self.clip(**tokens)
        emb = out.text_embeds  # [B, clip_dim]
        return emb / emb.norm(dim=-1, keepdim=True)

    def forward(self, queries):
        """queries: list[str] of length B -> [B, feat_ch] projected embedding."""
        emb = self._clip_embed(queries)
        return self.proj(emb)

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.device_ = device
        return self
