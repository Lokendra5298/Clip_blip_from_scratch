"""Phase 2 - text Transformer encoder from scratch.

Pipeline: token ids -> token embeddings + positional embeddings -> stack of
pre-norm Transformer blocks (with padding masked out) -> take the hidden state
at the <eos> position as the sentence summary (the same pooling CLIP uses).
"""
import torch
import torch.nn as nn

from .config import Config
from .layers import EncoderBlock, key_padding_bias


class TextTransformer(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, pad_id: int):
        super().__init__()
        dim = cfg.text_dim
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_length, dim))
        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, cfg.text_heads) for _ in range(cfg.text_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :L]
        bias = key_padding_bias(attention_mask)              # ignore padding keys
        for block in self.blocks:
            x = block(x, bias)
        x = self.norm(x)                                     # (B, L, dim)

        # The <eos> token is the last real token; attention_mask sums to its
        # index + 1, so subtract 1 to gather that position per sequence.
        eos_index = attention_mask.sum(dim=1) - 1            # (B,)
        pooled = x[torch.arange(B, device=x.device), eos_index]   # (B, dim)
        return x, pooled
