"""Central configuration for the CLIP/BLIP-from-scratch project.

A single dataclass holds every hyperparameter so there is exactly one place to
look and change things. Phase 1 only uses the data-related fields; the model and
training fields live here too so later phases reuse the same object.
"""
from dataclasses import dataclass
from pathlib import Path

# Repo root = parent of the `src/` directory. Deriving paths from __file__ means
# the scripts work no matter which directory you launch them from.
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- Data ----
    data_dir: Path = ROOT / "data" / "flickr8k"   # where Images/ + captions.txt live
    kaggle_dataset: str = "adityajn105/flickr8k"  # source dataset on Kaggle
    image_size: int = 128            # small on purpose -> fast epochs for the demo
    max_length: int = 32             # caption length cap in tokens (incl. <bos>/<eos>)
    min_word_freq: int = 2           # words rarer than this are mapped to <unk>

    # ---- Splits (done BY IMAGE, never by caption) ----
    n_val_images: int = 1000
    n_test_images: int = 1000
    split_seed: int = 42

    # ---- DataLoader ----
    batch_size: int = 64
    num_workers: int = 2

    # ---- Checkpoints (used from Phase 3) ----
    checkpoint_dir: Path = ROOT / "checkpoints"

    # ---- Vision encoder (used from Phase 2) ----
    patch_size: int = 16
    vision_dim: int = 256
    vision_layers: int = 6
    vision_heads: int = 8

    # ---- Text encoder (used from Phase 2) ----
    text_dim: int = 256
    text_layers: int = 6
    text_heads: int = 8

    # ---- Shared projection / training (used from Phase 2-3) ----
    projection_dim: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.1
    epochs: int = 30

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "Images"

    @property
    def captions_file(self) -> Path:
        return self.data_dir / "captions.txt"
