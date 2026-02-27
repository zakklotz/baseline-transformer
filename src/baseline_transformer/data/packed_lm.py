from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline_transformer.data.hf_datasets import load_lm_dataset


class PackedLMDataset(Dataset):
    """Packed token-stream dataset for causal LM training/eval.

    Steps:
    - load HF dataset split
    - tokenize each non-empty text row (no added special tokens)
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
    ):
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

        ds = load_lm_dataset(name, split)

        token_ids: list[int] = []
        for ex in ds:
            text = ex.get(text_column, "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                token_ids.extend(ids)

        if not token_ids:
            self.blocks = np.zeros((0, self.block_size), dtype=np.int64)
            return

        usable = (len(token_ids) // self.block_size) * self.block_size
        if usable == 0:
            self.blocks = np.zeros((0, self.block_size), dtype=np.int64)
            return

        stream = np.asarray(token_ids[:usable], dtype=np.int64)
        self.blocks = stream.reshape(-1, self.block_size)

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(self.blocks[idx]).to(torch.long)
        attention_mask = torch.ones(self.block_size, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
