"""Phase 5 - a simplified BLIP, built on the Phase 2 encoders.

BLIP trains three objectives over the same image features:
  1. ITC - image-text contrastive (identical to CLIP; reused here).
  2. ITM - image-text matching: a binary "do this image and caption match?"
           head on top of a text encoder that cross-attends to the image,
           trained with hard negatives mined from the ITC similarities.
  3. LM  - captioning: a causal text decoder that cross-attends to the image
           and is trained to predict the next token.

We deliberately leave out BLIP's CapFilt bootstrapping and the full shared-
parameter MED design - those add complexity without much pedagogical payoff at
Flickr8k scale. The blog notes them as "further reading".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import clip_contrastive_loss
from .config import Config
from .layers import GroundedBlock, causal_bias, key_padding_bias
from .text import TextTransformer
from .vision import VisionTransformer


class BLIP(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, pad_id: int):
        super().__init__()
        assert cfg.vision_dim == cfg.text_dim, \
            "this simplified BLIP keeps vision_dim == text_dim so cross-attention lines up"
        self.cfg = cfg
        self.pad_id = pad_id
        dim = cfg.text_dim

        # Shared image encoder (full token sequence is used for cross-attention).
        self.vision = VisionTransformer(cfg)

        # --- ITC head (contrastive), same design as CLIP ---
        self.text_encoder = TextTransformer(cfg, vocab_size, pad_id)
        self.visual_proj = nn.Linear(cfg.vision_dim, cfg.projection_dim, bias=False)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.projection_dim, bias=False)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        # --- ITM head (image-grounded text encoder + binary classifier) ---
        self.itm_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.itm_pos = nn.Parameter(torch.zeros(1, cfg.max_length, dim))
        self.itm_blocks = nn.ModuleList(
            [GroundedBlock(dim, cfg.text_heads, causal=False) for _ in range(2)]
        )
        self.itm_norm = nn.LayerNorm(dim)
        self.itm_head = nn.Linear(dim, 2)

        # --- LM head (causal image-grounded decoder for captioning) ---
        self.dec_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.dec_pos = nn.Parameter(torch.zeros(1, cfg.max_length, dim))
        self.dec_blocks = nn.ModuleList(
            [GroundedBlock(dim, cfg.text_heads, causal=True) for _ in range(cfg.text_layers)]
        )
        self.dec_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

        nn.init.trunc_normal_(self.itm_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

    # ---------- image ----------
    def encode_image(self, image):
        return self.vision(image)                       # (B, N+1, dim)

    # ---------- ITC ----------
    def itc(self, image_feats, input_ids, attention_mask):
        img = F.normalize(self.visual_proj(image_feats[:, 0]), dim=-1)
        _, pooled = self.text_encoder(input_ids, attention_mask)
        txt = F.normalize(self.text_proj(pooled), dim=-1)
        sims = img @ txt.t()                            # (B, B) cosine sims
        logits = self.log_temp.exp().clamp(max=100.0) * sims
        return clip_contrastive_loss(logits), sims

    # ---------- ITM ----------
    def _ground_encode(self, input_ids, attention_mask, image_feats):
        B, L = input_ids.shape
        x = self.itm_embed(input_ids) + self.itm_pos[:, :L]
        bias = key_padding_bias(attention_mask)
        for block in self.itm_blocks:
            x = block(x, image_feats, self_bias=bias)
        x = self.itm_norm(x)
        return x[:, 0]                                  # BOS position = multimodal [CLS]

    def itm(self, image_feats, input_ids, attention_mask, sims):
        B = input_ids.size(0)
        device = input_ids.device

        # For each image, the hardest negative caption = most similar mismatch.
        with torch.no_grad():
            neg = sims.clone()
            neg.fill_diagonal_(float("-inf"))
            neg_idx = neg.argmax(dim=1)                 # (B,)

        pos_cls = self._ground_encode(input_ids, attention_mask, image_feats)
        neg_cls = self._ground_encode(input_ids[neg_idx], attention_mask[neg_idx], image_feats)

        logits = self.itm_head(torch.cat([pos_cls, neg_cls], dim=0))   # (2B, 2)
        labels = torch.cat([torch.ones(B, dtype=torch.long, device=device),
                            torch.zeros(B, dtype=torch.long, device=device)])
        return F.cross_entropy(logits, labels)

    # ---------- LM (captioning) ----------
    def _decode_logits(self, input_ids, attention_mask, image_feats):
        B, L = input_ids.shape
        x = self.dec_embed(input_ids) + self.dec_pos[:, :L]
        self_bias = causal_bias(L, input_ids.device) + key_padding_bias(attention_mask)
        for block in self.dec_blocks:
            x = block(x, image_feats, self_bias=self_bias)
        x = self.dec_norm(x)
        return self.lm_head(x)                          # (B, L, vocab)

    def lm(self, image_feats, input_ids, attention_mask):
        # Teacher forcing: input is the caption minus its last token, the target
        # is the caption shifted left by one (predict the next token).
        dec_in, dec_mask = input_ids[:, :-1], attention_mask[:, :-1]
        target = input_ids[:, 1:]
        logits = self._decode_logits(dec_in, dec_mask, image_feats)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.pad_id,
        )

    def forward(self, image, input_ids, attention_mask):
        image_feats = self.encode_image(image)
        loss_itc, sims = self.itc(image_feats, input_ids, attention_mask)
        loss_itm = self.itm(image_feats, input_ids, attention_mask, sims)
        loss_lm = self.lm(image_feats, input_ids, attention_mask)
        return {
            "loss": loss_itc + loss_itm + loss_lm,
            "itc": loss_itc.item(),
            "itm": loss_itm.item(),
            "lm": loss_lm.item(),
        }

    @torch.no_grad()
    def generate(self, image, tokenizer, max_length=None, device="cpu"):
        """Greedy caption generation for a single image."""
        self.eval()
        max_length = max_length or self.cfg.max_length
        image = image.to(device)
        if image.dim() == 3:
            image = image.unsqueeze(0)                  # add batch dim
        image_feats = self.encode_image(image)

        ids = [tokenizer.bos_id]
        for _ in range(max_length - 1):
            inp = torch.tensor([ids], device=device)
            mask = torch.ones_like(inp)
            logits = self._decode_logits(inp, mask, image_feats)   # (1, len, vocab)
            next_id = int(logits[0, -1].argmax())
            if next_id == tokenizer.eos_id:
                break
            ids.append(next_id)
        return tokenizer.decode(ids)
