"""Phase 4 - evaluate a trained CLIP and produce the demos.

    python -m src.eval --task retrieval     # Recall@1/5/10 on the test split
    python -m src.eval --task zeroshot      # zero-shot CIFAR-10 accuracy
    python -m src.eval --task qualitative --query "a dog running on grass"

These are the figures/numbers that make the repo look serious in the README.
"""
import argparse

import torch
import torch.nn.functional as F

from .clip import CLIP
from .config import Config
from .dataset import build_transforms, build_dataloaders
from .tokenizer import SimpleTokenizer


def load_model(cfg, device):
    """Load the best checkpoint + the tokenizer saved alongside it."""
    ckpt_path = cfg.checkpoint_dir / "clip_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CLIP(ckpt["cfg"], ckpt["vocab_size"], ckpt["pad_id"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = SimpleTokenizer.load(cfg.checkpoint_dir / "tokenizer.json")
    return model, tokenizer


@torch.no_grad()
def _embed_split(model, loader, device):
    """Return aligned (image_embeddings, text_embeddings) over the whole split,
    using one caption per image (the loader's fixed eval caption)."""
    img_list, txt_list = [], []
    for batch in loader:
        img = model.encode_image(batch["image"].to(device))
        txt = model.encode_text(batch["input_ids"].to(device),
                                batch["attention_mask"].to(device))
        img_list.append(img.cpu())
        txt_list.append(txt.cpu())
    return torch.cat(img_list), torch.cat(txt_list)


def _recall_at_k(sim, ks=(1, 5, 10)):
    """sim: (N, N), matching pair is the diagonal. Returns recall for each k."""
    n = sim.size(0)
    targets = torch.arange(n)
    ranks = sim.argsort(dim=1, descending=True)          # (N, N) ranked indices
    out = {}
    for k in ks:
        topk = ranks[:, :k]
        hit = (topk == targets[:, None]).any(dim=1).float().mean().item()
        out[k] = 100.0 * hit
    return out


def evaluate_retrieval(cfg, device):
    model, _ = load_model(cfg, device)
    loaders, _ = build_dataloaders(cfg)
    img_emb, txt_emb = _embed_split(model, loaders["test"], device)
    sim = img_emb @ txt_emb.t()                          # cosine sims (unit vectors)

    i2t = _recall_at_k(sim)              # image -> text
    t2i = _recall_at_k(sim.t())          # text -> image
    print(f"Test images: {sim.size(0)} (one caption each)")
    print("Image -> Text   R@1 {:.1f}  R@5 {:.1f}  R@10 {:.1f}".format(i2t[1], i2t[5], i2t[10]))
    print("Text  -> Image  R@1 {:.1f}  R@5 {:.1f}  R@10 {:.1f}".format(t2i[1], t2i[5], t2i[10]))
    return i2t, t2i


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def evaluate_zeroshot(cfg, device, max_images=2000):
    """CLIP's signature trick: classify images it was never trained to label by
    comparing each image to text prompts like 'a photo of a {class}'."""
    from torchvision import datasets

    model, tokenizer = load_model(cfg, device)
    eval_tf = build_transforms(cfg.image_size, train=False)
    ds = datasets.CIFAR10(root=str(cfg.data_dir.parent / "cifar10"),
                          train=False, download=True, transform=eval_tf)

    # Encode the class prompts once.
    prompts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    ids, masks = zip(*[tokenizer.encode(p, cfg.max_length) for p in prompts])
    ids = torch.tensor(ids, device=device)
    masks = torch.tensor(masks, device=device)
    text_emb = model.encode_text(ids, masks)             # (10, d)

    correct, total = 0, 0
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size)
    for images, labels in loader:
        image_emb = model.encode_image(images.to(device))    # (B, d)
        preds = (image_emb @ text_emb.t()).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total >= max_images:
            break
    acc = 100.0 * correct / total
    print(f"Zero-shot CIFAR-10 accuracy on {total} images: {acc:.1f}%")
    return acc


@torch.no_grad()
def qualitative_retrieval(cfg, device, query, k=5):
    """Encode every test image, then return the top-k images for a text query."""
    model, tokenizer = load_model(cfg, device)
    loaders, _ = build_dataloaders(cfg)
    test = loaders["test"]

    img_emb, _ = _embed_split(model, test, device)       # (N, d), dataset order
    ids, mask = tokenizer.encode(query, cfg.max_length)
    q = model.encode_text(torch.tensor([ids], device=device),
                          torch.tensor([mask], device=device)).cpu()
    scores = (img_emb @ q.t()).squeeze(1)
    top = scores.topk(k).indices.tolist()

    names = [test.dataset.samples[i][0] for i in top]
    print(f'Top {k} images for "{query}":')
    for rank, name in enumerate(names, 1):
        print(f"  {rank}. {name}")
    _save_retrieval_figure(cfg, query, names)
    return names


def _save_retrieval_figure(cfg, query, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as exc:
        print(f"(skipped figure: {exc})")
        return

    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.imshow(Image.open(cfg.images_dir / name).convert("RGB"))
        ax.axis("off")
    fig.suptitle(f'query: "{query}"')
    out = cfg.checkpoint_dir / "retrieval_demo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved figure to {out}")


@torch.no_grad()
def plot_embedding_space(cfg, device, n=300):
    """Optional: t-SNE of image vs text embeddings to see the joint space."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(t-SNE needs scikit-learn + matplotlib: {exc})")
        return

    model, _ = load_model(cfg, device)
    loaders, _ = build_dataloaders(cfg)
    img_emb, txt_emb = _embed_split(model, loaders["test"], device)
    img_emb, txt_emb = img_emb[:n], txt_emb[:n]

    pts = torch.cat([img_emb, txt_emb]).numpy()
    emb2d = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(pts)
    m = img_emb.size(0)
    plt.figure(figsize=(6, 6))
    plt.scatter(emb2d[:m, 0], emb2d[:m, 1], s=8, label="image")
    plt.scatter(emb2d[m:, 0], emb2d[m:, 1], s=8, label="text")
    plt.legend()
    out = cfg.checkpoint_dir / "embedding_tsne.png"
    plt.savefig(out, dpi=120)
    print(f"Saved t-SNE to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["retrieval", "zeroshot", "qualitative", "tsne"],
                        default="retrieval")
    parser.add_argument("--query", default="a dog running on grass")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = Config()
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    if args.task == "retrieval":
        evaluate_retrieval(cfg, device)
    elif args.task == "zeroshot":
        evaluate_zeroshot(cfg, device)
    elif args.task == "qualitative":
        qualitative_retrieval(cfg, device, args.query)
    elif args.task == "tsne":
        plot_embedding_space(cfg, device)


if __name__ == "__main__":
    main()
