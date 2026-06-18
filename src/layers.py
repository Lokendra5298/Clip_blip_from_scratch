"""Phase 2 - shared Transformer building blocks, implemented from scratch.

The vision encoder, the text encoder, and (in Phase 5) the BLIP grounded
encoder/decoder are all built from the same pieces: multi-head attention done
by hand, a small MLP, and pre-norm residual blocks. Keeping them here means the
attention math lives in exactly one readable place.
"""
import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, attn_bias=None):
    """The core operation.

    q, k, v : (B, H, L, d)
    attn_bias : additive mask broadcastable to (B, H, Lq, Lk); use -inf to
                forbid a query->key edge (it becomes 0 after softmax).
    """
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)   # (B, H, Lq, Lk)
    if attn_bias is not None:
        scores = scores + attn_bias
    weights = scores.softmax(dim=-1)
    return weights @ v                                   # (B, H, Lq, d)


class MultiHeadAttention(nn.Module):
    """Multi-head attention. Used for self-attention (pass x as both q and kv)
    and, in Phase 5, cross-attention (q = text, kv = image features)."""

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,L,d)

    def forward(self, x_q, x_kv, attn_bias=None):
        q = self._split_heads(self.q_proj(x_q))
        k = self._split_heads(self.k_proj(x_kv))
        v = self._split_heads(self.v_proj(x_kv))

        out = scaled_dot_product_attention(q, k, v, attn_bias)  # (B,H,Lq,d)
        B, H, Lq, d = out.shape
        out = out.transpose(1, 2).reshape(B, Lq, H * d)         # merge heads
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm self-attention block, shared by the vision and text encoders."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, attn_bias=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, attn_bias)   # residual around attention
        x = x + self.ff(self.norm2(x))       # residual around MLP
        return x


class GroundedBlock(nn.Module):
    """Phase 5 (BLIP): self-attention + cross-attention to image features + MLP.

    causal=False -> image-grounded text *encoder* (used for image-text matching)
    causal=True  -> image-grounded text *decoder* (used for captioning)
    """

    def __init__(self, dim, n_heads, causal):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, image_feats, self_bias=None):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, self_bias)
        h = self.norm2(x)
        x = x + self.cross_attn(h, image_feats, None)  # attend over all image tokens
        x = x + self.ff(self.norm3(x))
        return x


def key_padding_bias(attention_mask):
    """attention_mask: (B, L) with 1=keep, 0=pad. Returns additive bias (B,1,1,L)
    that sets padding *keys* to -inf so no query attends to them."""
    bias = torch.zeros_like(attention_mask, dtype=torch.float)
    bias = bias.masked_fill(attention_mask == 0, float("-inf"))
    return bias[:, None, None, :]


def causal_bias(L, device):
    """(1,1,L,L) additive mask so position i cannot attend to positions > i."""
    forbidden = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    bias = torch.zeros(L, L, device=device).masked_fill(forbidden, float("-inf"))
    return bias[None, None, :, :]
