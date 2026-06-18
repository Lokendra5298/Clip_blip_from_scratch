"""Phase 1 sanity check - run this right after downloading the data.

    python -m src.sanity_check

It builds the dataloaders, pulls one training batch, prints tensor shapes and a
few decoded captions, and (if matplotlib is available) saves a small grid of
images with their captions to data/sanity_batch.png so you can eyeball that the
images and captions actually line up. Catching mismatches here saves hours of
debugging a model that "trains" on garbage later.
"""
from .config import Config
from .dataset import build_dataloaders


def main():
    cfg = Config()
    loaders, tokenizer = build_dataloaders(cfg)

    print(f"vocab size: {tokenizer.vocab_size}")
    for split, loader in loaders.items():
        print(f"{split:>5}: {len(loader.dataset):>5} images, {len(loader):>4} batches")

    batch = next(iter(loaders["train"]))
    print("\nOne training batch:")
    print("  image          :", tuple(batch["image"].shape))
    print("  input_ids      :", tuple(batch["input_ids"].shape))
    print("  attention_mask :", tuple(batch["attention_mask"].shape))

    print("\nFirst few decoded captions in the batch:")
    for i in range(min(4, batch["input_ids"].shape[0])):
        print("  -", tokenizer.decode(batch["input_ids"][i].tolist()))

    try:
        _save_grid(cfg, batch, tokenizer)
    except Exception as exc:  # matplotlib missing or headless issue - not fatal
        print(f"\n(skipped image grid: {exc})")


def _save_grid(cfg, batch, tokenizer, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n = min(n, batch["image"].shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = (batch["image"][i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        caption = tokenizer.decode(batch["input_ids"][i].tolist())
        axes[i].imshow(img)
        axes[i].set_title(caption, fontsize=8, wrap=True)
        axes[i].axis("off")

    out = cfg.data_dir.parent / "sanity_batch.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved batch preview to {out}")


if __name__ == "__main__":
    main()
