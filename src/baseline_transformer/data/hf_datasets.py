from __future__ import annotations

from datasets import load_dataset


def load_lm_dataset(name: str, split: str, config_name: str | None = None):
    # Default: preserve the historical raw WikiText-103 baseline unless explicitly overridden.
    if name == "wikitext103":
        ds = load_dataset("wikitext", config_name or "wikitext-103-raw-v1", split=split)
        return ds
    if config_name is not None:
        return load_dataset(name, config_name, split=split)
    return load_dataset(name, split=split)
