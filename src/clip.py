"""Phase 2 - CLIP: tie the two encoders together with a contrastive objective.

Both encoders project into a shared, L2-normalized embedding space. Training
pulls matching image-text pairs together and pushes mismatched pairs apart. The
loss is the heart of the project and is derived line-by-line in the blog.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .text import TextTransformer
from .vision import VisionTransformer


class CLIP(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, pad_id: int):
        super().__init__()
        self.vision = VisionTransformer(cfg)
        self.text = TextTransformer(cfg, vocab_size, pad_id)
        # No-bias linear heads map each encoder into the shared space.
        self.visual_proj = nn.Linear(cfg.vision_dim, cfg.projection_dim, bias=False)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.projection_dim, bias=False)
        # Learnable temperature, stored in log-space and clamped (as in CLIP).
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, image):
        cls = self.vision(image)[:, 0]                       # (B, vision_dim)
        return F.normalize(self.visual_proj(cls), dim=-1)    # (B, proj_dim), unit norm

    def encode_text(self, input_ids, attention_mask):
        _, pooled = self.text(input_ids, attention_mask)     # (B, text_dim)
        return F.normalize(self.text_proj(pooled), dim=-1)   # (B, proj_dim), unit norm

    def forward(self, image, input_ids, attention_mask):
        image_emb = self.encode_image(image)                 # (B, d)
        text_emb = self.encode_text(input_ids, attention_mask)
        logit_scale = self.log_temp.exp().clamp(max=100.0)
        # Because both are unit vectors, the dot product is cosine similarity.
        logits = logit_scale * image_emb @ text_emb.t()      # (B, B): rows=images
        return logits


def clip_contrastive_loss(logits):
    """Symmetric InfoNCE.

    The B matching pairs sit on the diagonal of the (B, B) similarity matrix.
    We want each row (an image) to rank its own caption highest and each column
    (a caption) to rank its own image highest. That is exactly cross-entropy
    against the labels [0, 1, ..., B-1] applied along both axes, then averaged.
    """
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_image_to_text = F.cross_entropy(logits, targets)
    loss_text_to_image = F.cross_entropy(logits.t(), targets)
    return (loss_image_to_text + loss_text_to_image) / 2


@torch.no_grad()
def retrieval_accuracy(logits):
    """In-batch top-1 accuracy: how often the diagonal wins its row / column.
    A cheap signal to watch during training (returns image->text, text->image)."""
    targets = torch.arange(logits.size(0), device=logits.device)
    i2t = (logits.argmax(dim=1) == targets).float().mean().item()
    t2i = (logits.argmax(dim=0) == targets).float().mean().item()
    return i2t, t2i
