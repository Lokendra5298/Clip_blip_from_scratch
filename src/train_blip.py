"""Phase 5 - train the simplified BLIP on Flickr8k.

    python -m src.train_blip --epochs 30
    python -m src.train_blip --overfit 200     # debug: overfit ONE batch

Reuses the schedule / optimizer / device helpers from the CLIP trainer so the
two training scripts stay consistent. The total loss is ITC + ITM + LM; we log
each component so you can see which objective is (or isn't) learning.
"""
import argparse

import torch

from .blip import BLIP
from .config import Config
from .dataset import build_dataloaders
from .train import build_optimizer, cosine_warmup, move_batch, resolve_device


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], batch["input_ids"], batch["attention_mask"])
        total += out["loss"].item()
        n += 1
    return total / max(1, n)


def overfit_one_batch(model, loader, optimizer, device, steps):
    model.train()
    batch = move_batch(next(iter(loader)), device)
    for step in range(1, steps + 1):
        out = model(batch["image"], batch["input_ids"], batch["attention_mask"])
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0 or step == 1:
            print(f"  step {step:>4} | total {out['loss'].item():.3f} "
                  f"| itc {out['itc']:.3f} itm {out['itm']:.3f} lm {out['lm']:.3f}")


def train(cfg, device, epochs, use_amp):
    loaders, tokenizer = build_dataloaders(cfg)
    model = BLIP(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
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
                out = model(batch["image"], batch["input_ids"], batch["attention_mask"])
                loss = out["loss"]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(loaders['train'])} "
                      f"loss {running / i:.3f} | itc {out['itc']:.3f} "
                      f"itm {out['itm']:.3f} lm {out['lm']:.3f}")

        val_loss = evaluate_loss(model, loaders["val"], device)
        print(f"== epoch {epoch}: train {running / len(loaders['train']):.3f} | val {val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "vocab_size": tokenizer.vocab_size,
                 "pad_id": tokenizer.pad_id, "cfg": cfg},
                cfg.checkpoint_dir / "blip_best.pt",
            )
            print(f"   saved new best to {cfg.checkpoint_dir / 'blip_best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overfit", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.overfit > 0:
        loaders, tokenizer = build_dataloaders(cfg)
        model = BLIP(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
        optimizer = build_optimizer(model, cfg)
        overfit_one_batch(model, loaders["train"], optimizer, device, args.overfit)
    else:
        train(cfg, device, args.epochs, args.amp)


if __name__ == "__main__":
    main()
