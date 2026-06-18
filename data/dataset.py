"""Phase 1 - Flickr8k Dataset, splits, transforms, and DataLoaders.

captions.txt has a header line `image,caption` followed by 5 rows per image.
The pipeline:
  1. group captions by image filename,
  2. split BY IMAGE so no image leaks across train/val/test,
  3. build the tokenizer from the TRAIN captions only,
  4. return (image_tensor, input_ids, attention_mask) for each item.

For training we pick one of the 5 captions at random each step (cheap text
augmentation); for val/test we use a fixed caption so results are reproducible.
"""
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config
from .tokenizer import SimpleTokenizer

# ImageNet normalization stats - standard choice, also handy later if you swap
# in a pretrained vision backbone.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def load_captions(captions_file: Path) -> Dict[str, List[str]]:
    """Read captions.txt into {image_filename: [caption, ...]}."""
    image_to_caps: Dict[str, List[str]] = {}
    with open(captions_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the "image,caption" header
        for row in reader:
            if len(row) < 2:
                continue
            name, caption = row[0].strip(), row[1].strip()
            image_to_caps.setdefault(name, []).append(caption)
    return image_to_caps


def split_by_image(
    image_to_caps: Dict[str, List[str]], cfg: Config
) -> Tuple[list, list, list]:
    """Shuffle image names with a fixed seed and carve out val/test/train."""
    names = sorted(image_to_caps.keys())
    rng = random.Random(cfg.split_seed)
    rng.shuffle(names)

    test_names = names[: cfg.n_test_images]
    val_names = names[cfg.n_test_images : cfg.n_test_images + cfg.n_val_images]
    train_names = names[cfg.n_test_images + cfg.n_val_images :]

    def to_pairs(ns):
        return [(n, image_to_caps[n]) for n in ns]

    return to_pairs(train_names), to_pairs(val_names), to_pairs(test_names)


def build_transforms(image_size: int, train: bool):
    """Train transforms add light augmentation; eval transforms are deterministic."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


class Flickr8kDataset(Dataset):
    def __init__(self, samples, images_dir, tokenizer, cfg: Config, train: bool):
        self.samples = samples              # list of (image_name, [captions])
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.train = train
        self.transform = build_transforms(cfg.image_size, train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, captions = self.samples[idx]
        image = Image.open(self.images_dir / name).convert("RGB")
        image = self.transform(image)

        caption = random.choice(captions) if self.train else captions[0]
        ids, attn = self.tokenizer.encode(caption, self.max_length)
        return {
            "image": image,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def build_dataloaders(cfg: Config):
    """Top-level helper: returns ({'train','val','test': DataLoader}, tokenizer)."""
    image_to_caps = load_captions(cfg.captions_file)
    train_s, val_s, test_s = split_by_image(image_to_caps, cfg)

    # Build the tokenizer from TRAIN captions only to avoid val/test leakage.
    train_captions = [c for _, caps in train_s for c in caps]
    tokenizer = SimpleTokenizer.build(train_captions, cfg.min_word_freq)

    def make(samples, train):
        ds = Flickr8kDataset(samples, cfg.images_dir, tokenizer, cfg, train)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.num_workers,
            drop_last=train,
        )

    loaders = {
        "train": make(train_s, True),
        "val": make(val_s, False),
        "test": make(test_s, False),
    }
    return loaders, tokenizer
