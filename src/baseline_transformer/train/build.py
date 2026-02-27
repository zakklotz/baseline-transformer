from __future__ import annotations

import os
from torch.utils.data import DataLoader

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.nncore_bridge import build_transformer_config
from baseline_transformer.models import StandardTransformerLM, RecursiveTransformerLM

from baseline_transformer.data import (
    load_lm_dataset,
    get_tokenizer,
    CausalLMCollator,
)

# Optional packed dataset support (only if your repo includes it)
try:
    from baseline_transformer.data.packed_lm import PackedLMDataset  # type: ignore
except Exception:  # pragma: no cover
    PackedLMDataset = None  # type: ignore


def _is_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def build_everything(cfg: ExperimentConfig):
    """
    Build model, dataloaders, and tokenizer.

    Returns:
      model, train_loader, val_loader, tokenizer
    """
    # --- Tokenizer ---
    tok = get_tokenizer(cfg.data.get("tokenizer", "gpt2"))

    # --- DataLoader workers ---
    # Prefer config-driven, but force 0 under pytest to avoid fork warnings
    num_workers = int(cfg.data.get("num_workers", 0))
    if _is_pytest():
        num_workers = 0

    batch_size = int(cfg.train["batch_size"])
    max_seq_len = int(cfg.data.get("max_seq_len", 512))

    packing = bool(cfg.data.get("packing", False))
    block_size = int(cfg.data.get("block_size", max_seq_len))

    # Per your earlier improvements: interpret data.stride as eval stride by default
    eval_stride = cfg.data.get("stride", None)
    eval_stride = int(eval_stride) if eval_stride is not None else None
    train_stride = cfg.data.get("train_stride", None)
    train_stride = int(train_stride) if train_stride is not None else None

    if packing:
        if PackedLMDataset is None:
            raise RuntimeError(
                "data.packing=true but PackedLMDataset is not available. "
                "Ensure baseline_transformer/data/packed_lm.py exists."
            )

        # Build packed datasets
        train_base = load_lm_dataset(cfg.data["name"], cfg.data.get("split_train", "train"))
        val_base = load_lm_dataset(cfg.data["name"], cfg.data.get("split_val", "validation"))

        train_ds = PackedLMDataset(
            dataset=train_base,
            tokenizer=tok,
            block_size=block_size,
            stride=train_stride,   # default None -> non-overlapping for training
        )
        val_ds = PackedLMDataset(
            dataset=val_base,
            tokenizer=tok,
            block_size=block_size,
            stride=eval_stride,    # optional overlap for eval if configured
        )

        # Packed mode yields fixed-size tensors already; default collate is fine.
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
        # Classic per-row tokenization path
        train_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_train", "train"))
        val_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_val", "validation"))

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

        # Force unambiguous recursive baseline semantics:
        # one shared decoder block repeated `depth` times via recurrence.
        tcfg.num_encoder_layers = 0
        tcfg.num_decoder_layers = 1
        tcfg.recursive = True
        tcfg.recurrence_steps = depth

        model = RecursiveTransformerLM(tcfg, depth=depth)

    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    return model, train_loader, val_loader, tok
