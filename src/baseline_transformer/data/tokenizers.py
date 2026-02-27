from __future__ import annotations

from transformers import AutoTokenizer


def get_tokenizer(name: str):
    if name == "gpt2":
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        # GPT-2 has no pad token by default; set it to eos for batching.
        tok.pad_token = tok.eos_token
        return tok
    # Extendable
    return AutoTokenizer.from_pretrained(name, use_fast=True)
