from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.data import load_lm_dataset, get_tokenizer, CausalLMCollator
from baseline_transformer.models import StandardTransformerLM, RecursiveTransformerLM
from baseline_transformer.nncore_bridge import build_transformer_config


def build_everything(cfg: ExperimentConfig):
    # Tokenizer + datasets
    tok = get_tokenizer(cfg.data.get("tokenizer", "gpt2"))
    train_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_train", "train"))
    val_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_val", "validation"))

    collate = CausalLMCollator(tokenizer=tok, max_seq_len=int(cfg.data["max_seq_len"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )

    # Model
    tcfg = build_transformer_config(cfg.model)
    model_type = cfg.model.get("type", "standard_transformer")

    if model_type == "standard_transformer":
        model = StandardTransformerLM(tcfg)
    elif model_type == "recursive_transformer":
        depth = int(cfg.model.get("depth", tcfg.n_layers))
        model = RecursiveTransformerLM(tcfg, depth=depth)
    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    return model, train_loader, val_loader, tok
