from __future__ import annotations

import os
from torch.utils.data import DataLoader

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.nncore_bridge import build_transformer_config
from baseline_transformer.models import StandardTransformerLM, RecursiveTransformerLM

from baseline_transformer.data import get_tokenizer, CausalLMCollator, load_lm_dataset
from baseline_transformer.data.packed_lm import PackedLMDataset


def _is_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def build_everything(cfg: ExperimentConfig):
    """
    Build model, train/val dataloaders, and tokenizer.

    Returns:
      model, train_loader, val_loader, tokenizer
    """
    # --- Tokenizer ---
    tok = get_tokenizer(cfg.data.get("tokenizer", "gpt2"))

    # --- DataLoader workers ---
    num_workers = int(cfg.data.get("num_workers", 0))
    if _is_pytest():
        num_workers = 0

    batch_size = int(cfg.train["batch_size"])
    max_seq_len = int(cfg.data.get("max_seq_len", 512))

    # --- Data mode ---
    packing = bool(cfg.data.get("packing", False))
    block_size = int(cfg.data.get("block_size", max_seq_len))
    text_column = str(cfg.data.get("text_column", "text"))

    split_train = str(cfg.data.get("split_train", "train"))
    split_val = str(cfg.data.get("split_val", "validation"))
    ds_name = str(cfg.data["name"])
    ds_config = cfg.data.get("dataset_config")
    ds_config = str(ds_config) if ds_config is not None else None

    # Interpret data.stride as eval stride by default (safer comparisons)
    eval_stride = cfg.data.get("stride", None)
    eval_stride = int(eval_stride) if eval_stride is not None else None

    # Allow overriding for training if desired
    train_stride = cfg.data.get("train_stride", None)
    train_stride = int(train_stride) if train_stride is not None else None

    if packing:
        # Packed dataset loads HF split internally
        train_ds = PackedLMDataset(
            ds_name,
            split_train,
            tok,
            block_size=block_size,
            text_column=text_column,
            dataset_config=ds_config,
            stride=train_stride,  # default None -> non-overlapping for train
        )
        val_ds = PackedLMDataset(
            ds_name,
            split_val,
            tok,
            block_size=block_size,
            text_column=text_column,
            dataset_config=ds_config,
            stride=eval_stride,   # optional overlap for eval
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    else:
        # Per-row tokenization path (HF dataset object + collator)
        train_ds = load_lm_dataset(ds_name, split_train, config_name=ds_config)
        val_ds = load_lm_dataset(ds_name, split_val, config_name=ds_config)

        collate = CausalLMCollator(tokenizer=tok, max_seq_len=max_seq_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

    # --- Model config (nn-core TransformerConfig) ---
    tcfg = build_transformer_config(cfg.model)
    model_type = cfg.model.get("type", "standard_transformer")

    if model_type == "standard_transformer":
        model = StandardTransformerLM(tcfg)

    elif model_type == "recursive_transformer":
        depth = int(cfg.model.get("depth", 6))

        # Force unambiguous semantics: 1 shared decoder block repeated depth times
        tcfg.num_encoder_layers = 0
        tcfg.num_decoder_layers = 1
        tcfg.recursive = True
        tcfg.recurrence_steps = depth

        model = RecursiveTransformerLM(tcfg, depth=depth)

    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    return model, train_loader, val_loader, tok
