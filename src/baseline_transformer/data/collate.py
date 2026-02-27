from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class CausalLMCollator:
    tokenizer: Any
    max_seq_len: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Expect HF dataset records with "text"
        texts = [ex.get("text", "") for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # Standard causal LM: labels are next-token prediction on same ids
        labels = input_ids.clone()
        labels[attn == 0] = -100  # ignore padding

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
