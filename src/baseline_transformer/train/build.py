from __future__ import annotations

import os

from torch.utils.data import DataLoader

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.data import CausalLMCollator, get_tokenizer, load_lm_dataset
from baseline_transformer.models import StandardTransformerLM
from baseline_transformer.nncore_bridge import build_transformer_config


def build_everything(cfg: ExperimentConfig):
    # Tokenizer + datasets
    tok = get_tokenizer(cfg.data.get("tokenizer", "gpt2"))
    train_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_train", "train"))
    val_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_val", "validation"))

    collate = CausalLMCollator(tokenizer=tok, max_seq_len=int(cfg.data["max_seq_len"]))

    in_pytest = "PYTEST_CURRENT_TEST" in os.environ
    num_workers = 0 if in_pytest else int(cfg.train.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    # Model
    tcfg = build_transformer_config(cfg.model)
    model_type = cfg.model.get("type", "standard_transformer")

    if model_type == "standard_transformer":
        model = StandardTransformerLM(tcfg)

    elif model_type == "recursive_transformer":
        depth = int(cfg.model.get("depth", 1))
        tcfg.recursive = True
        tcfg.recurrence_steps = depth
        tcfg.num_decoder_layers = 1
        model = StandardTransformerLM(tcfg)

    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    return model, train_loader, val_loader, tok
