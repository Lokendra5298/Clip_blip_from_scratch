"""Phase 2 - Vision Transformer (ViT) from scratch.

Pipeline: image -> non-overlapping patches -> linear patch embeddings ->
prepend a learnable [CLS] token -> add learnable positional embeddings ->
stack of pre-norm Transformer blocks. The [CLS] output is the image summary
CLIP uses; the full token sequence is what BLIP cross-attends to in Phase 5.
"""
import torch
import torch.nn as nn

from .config import Config
from .layers import EncoderBlock


class PatchEmbed(nn.Module):
    """Cut the image into patch_size x patch_size patches and embed each one.
    A strided convolution does both the cutting and the linear projection."""

    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                    # x: (B, 3, H, W)
        x = self.proj(x)                     # (B, dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)     # (B, n_patches, dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        dim = cfg.vision_dim
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, dim)
        n_tokens = self.patch_embed.n_patches + 1            # +1 for the CLS token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, cfg.vision_heads) for _ in range(cfg.vision_layers)]
        )
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image):
        B = image.shape[0]
        x = self.patch_embed(image)                          # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)                       # (B, N+1, dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x                                             # x[:, 0] is the CLS summary
