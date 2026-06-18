"""Phase 1 - download Flickr8k.

Flickr8k is ~1 GB: about 8,000 images, each paired with 5 human-written
captions. We pull it from Kaggle with `kagglehub` and copy it into
./data/flickr8k so the rest of the code can rely on a fixed location.

Run:
    python -m src.download

Prerequisites:
    pip install kagglehub
    Kaggle API credentials at ~/.kaggle/kaggle.json
    (see https://www.kaggle.com/docs/api)
"""
import shutil
from pathlib import Path

from .config import Config


def already_downloaded(cfg: Config) -> bool:
    return cfg.images_dir.is_dir() and cfg.captions_file.is_file()


def download(cfg: Config) -> Path:
    if already_downloaded(cfg):
        print(f"Data already present at {cfg.data_dir} - nothing to do.")
        return cfg.data_dir

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("kagglehub is not installed. Run: pip install kagglehub") from exc

    print(f"Downloading '{cfg.kaggle_dataset}' from Kaggle ...")
    src = Path(kagglehub.dataset_download(cfg.kaggle_dataset))
    print(f"Downloaded to Kaggle cache: {src}")

    # The mirror stores everything at its root, but we search recursively to be
    # robust to small layout differences between Flickr8k mirrors.
    images_src = _find(src, "Images", want_dir=True)
    captions_src = _find(src, "captions.txt", want_dir=False)
    if images_src is None or captions_src is None:
        raise SystemExit(
            f"Could not locate Images/ and captions.txt under {src}.\n"
            "The Kaggle mirror layout may differ - inspect the folder above "
            "and point Config.data_dir at it directly."
        )

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying dataset into {cfg.data_dir} ...")
    if not cfg.images_dir.exists():
        shutil.copytree(images_src, cfg.images_dir)
    shutil.copy(captions_src, cfg.captions_file)
    print("Done.")
    return cfg.data_dir


def _find(root: Path, name: str, want_dir: bool):
    for path in root.rglob(name):
        if path.is_dir() == want_dir:
            return path
    return None


if __name__ == "__main__":
    cfg = Config()
    download(cfg)
    n_images = len(list(cfg.images_dir.glob("*.jpg")))
    print(f"{n_images} images available under {cfg.images_dir}")
