from __future__ import annotations

from datasets import load_dataset


def load_lm_dataset(name: str, split: str):
    # Default: WikiText-103 (raw text)
    if name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        return ds
    # Extendable
    return load_dataset(name, split=split)
