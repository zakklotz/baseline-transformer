from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import Dataset

from baseline_transformer.data.hf_datasets import load_lm_dataset


def _tqdm(iterable, *, total=None, desc=None):
    """Optional tqdm wrapper (no hard dependency)."""
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


class PackedLMDataset(Dataset):
    """Packed token-stream dataset for causal LM training/eval.

    Steps:
    - load HF dataset split (or use provided in-memory texts)
    - tokenize each non-empty text row (no added special tokens)
    - optionally append eos token separator between rows
    - concatenate token ids into one stream
    - split into contiguous fixed-size blocks
    """

    def __init__(
        self,
        name: str,
        split: str,
        tokenizer: Any,
        block_size: int = 512,
        text_column: str = "text",
        dataset_config: str | None = None,
        stride: int | None = None,
        texts: Iterable[str] | None = None,
    ):
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

        step = self.block_size if stride is None else int(stride)
        if step <= 0:
            raise ValueError("stride must be > 0 when provided")

        eos_id = getattr(tokenizer, "eos_token_id", None)

        token_ids: list[int] = []

        if texts is None:
            ds = load_lm_dataset(name, split, config_name=dataset_config)
            total = len(ds) if hasattr(ds, "__len__") else None
            it = _tqdm(ds, total=total, desc=f"Packing {name}:{split}")
            for ex in it:
                text = ex.get(text_column, "")
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                ids = _encode_without_length_warning(tokenizer, text)
                if ids:
                    token_ids.extend(ids)
                    if eos_id is not None:
                        token_ids.append(int(eos_id))
        else:
            it = _tqdm(texts, desc="Packing texts")
            for text in it:
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                ids = _encode_without_length_warning(tokenizer, text)
                if ids:
                    token_ids.extend(ids)
                    if eos_id is not None:
                        token_ids.append(int(eos_id))

        self.stream = torch.tensor(token_ids, dtype=torch.long)

        if self.stream.numel() < self.block_size:
            self.starts: list[int] = []
            return

        max_start = self.stream.numel() - self.block_size
        self.starts = list(range(0, max_start + 1, step))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        input_ids = self.stream[start : start + self.block_size]
        attention_mask = torch.ones(self.block_size, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _encode_without_length_warning(tokenizer: Any, text: str) -> list[int]:
    """Encode text without triggering model_max_length warnings during packing."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            max_length=None,
            verbose=False,
        )
        ids = encoded["input_ids"]
        return list(ids)
    except Exception:
        return tokenizer.encode(text, add_special_tokens=False)
