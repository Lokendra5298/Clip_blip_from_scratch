"""Phase 3 - train CLIP on Flickr8k.

    python -m src.train --epochs 30           # full training
    python -m src.train --overfit 200         # debug: overfit ONE batch

The overfit mode is the recommended first run: if the model cannot drive the
loss on a single batch to near zero, the loss/masking is wrong and there is no
point training on the full set yet.

Utilities here (cosine schedule, optimizer, batch move) are reused by the BLIP
trainer in Phase 5.
"""
import argparse
import math

import torch

from .clip import CLIP, clip_contrastive_loss, retrieval_accuracy
from .config import Config
from .dataset import build_dataloaders


def cosine_warmup(step, warmup_steps, total_steps):
    """LR multiplier: linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["image"], batch["input_ids"], batch["attention_mask"])
        total += clip_contrastive_loss(logits).item()
        n += 1
    return total / max(1, n)


def overfit_one_batch(model, loader, optimizer, device, steps):
    """Sanity ritual: repeatedly fit a single batch; loss should fall to ~0."""
    model.train()
    batch = move_batch(next(iter(loader)), device)
    for step in range(1, steps + 1):
        logits = model(batch["image"], batch["input_ids"], batch["attention_mask"])
        loss = clip_contrastive_loss(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0 or step == 1:
            i2t, t2i = retrieval_accuracy(logits)
            print(f"  step {step:>4} | loss {loss.item():.4f} | i2t {i2t:.2f} t2i {t2i:.2f}")
    print("If loss approached 0 and accuracies approached 1.0, the model is wired correctly.")


def train(cfg, device, epochs, use_amp):
    loaders, tokenizer = build_dataloaders(cfg)
    model = CLIP(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    optimizer = build_optimizer(model, cfg)

    total_steps = epochs * len(loaders["train"])
    warmup_steps = max(1, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_warmup(s, warmup_steps, total_steps)
    )
    amp_on = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_on)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.checkpoint_dir / "tokenizer.json")
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for i, batch in enumerate(loaders["train"], 1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_on):
                logits = model(batch["image"], batch["input_ids"], batch["attention_mask"])
                loss = clip_contrastive_loss(logits)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(loaders['train'])} "
                      f"loss {running / i:.4f} lr {scheduler.get_last_lr()[0]:.2e}")

        val_loss = evaluate_loss(model, loaders["val"], device)
        print(f"== epoch {epoch}: train {running / len(loaders['train']):.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "pad_id": tokenizer.pad_id,
                "cfg": cfg,
            }
            torch.save(ckpt, cfg.checkpoint_dir / "clip_best.pt")
            print(f"   saved new best to {cfg.checkpoint_dir / 'clip_best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--overfit", type=int, default=0,
                        help="if > 0, overfit a single batch for this many steps and exit")
    args = parser.parse_args()

    cfg = Config()
    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.overfit > 0:
        loaders, tokenizer = build_dataloaders(cfg)
        model = CLIP(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
        optimizer = build_optimizer(model, cfg)
        overfit_one_batch(model, loaders["train"], optimizer, device, args.overfit)
    else:
        train(cfg, device, args.epochs, args.amp)


if __name__ == "__main__":
    main()
