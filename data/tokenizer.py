"""Phase 1 - a tiny word-level tokenizer built from the Flickr8k captions.

We deliberately avoid BPE / subword tokenizers to keep things readable: the
Flickr8k vocabulary is only a few thousand words. The tokenizer maps words to
integer ids, wraps each caption with <bos>/<eos>, and pads/truncates to a fixed
length so every sample has the same shape (no custom collate function needed).
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import List

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]  # PAD must stay at index 0


def tokenize(text: str) -> List[str]:
    """Lowercase and split into word tokens. Simple, but enough for captions."""
    return re.findall(r"\w+", text.lower())


class SimpleTokenizer:
    def __init__(self, stoi: dict):
        self.stoi = stoi                                   # string -> id
        self.itos = {i: s for s, i in stoi.items()}        # id -> string
        self.pad_id = stoi[PAD]
        self.bos_id = stoi[BOS]
        self.eos_id = stoi[EOS]
        self.unk_id = stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(cls, captions: List[str], min_freq: int = 2) -> "SimpleTokenizer":
        """Build a vocabulary from a list of caption strings."""
        counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))

        stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for word, freq in counter.most_common():
            if freq >= min_freq:
                stoi[word] = len(stoi)
        return cls(stoi)

    def encode(self, text: str, max_length: int):
        """Return (input_ids, attention_mask), both length == max_length."""
        ids = [self.stoi.get(w, self.unk_id) for w in tokenize(text)]
        ids = ids[: max_length - 2]                        # leave room for bos/eos
        ids = [self.bos_id] + ids + [self.eos_id]
        attn = [1] * len(ids)

        pad_n = max_length - len(ids)                      # right-pad to fixed length
        ids += [self.pad_id] * pad_n
        attn += [0] * pad_n
        return ids, attn

    def decode(self, ids: List[int]) -> str:
        """Turn ids back into text, dropping special tokens (stop at <eos>)."""
        words = []
        for i in ids:
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            words.append(self.itos.get(i, UNK))
        return " ".join(words)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.stoi), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SimpleTokenizer":
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi)
