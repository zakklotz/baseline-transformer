from __future__ import annotations

from torch.utils.data import DataLoader

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.data import load_lm_dataset, get_tokenizer, CausalLMCollator
from baseline_transformer.models import StandardTransformerLM
from baseline_transformer.nncore_bridge import build_transformer_config


def build_everything(cfg: ExperimentConfig):
    # Tokenizer + datasets
    tok = get_tokenizer(cfg.data.get("tokenizer", "gpt2"))
    train_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_train", "train"))
    val_ds = load_lm_dataset(cfg.data["name"], cfg.data.get("split_val", "validation"))

    collate = CausalLMCollator(tokenizer=tok, max_seq_len=int(cfg.data["max_seq_len"]))

    # NOTE: num_workers=0 avoids fork() warnings/deadlocks during pytest runs
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    # Model
    tcfg = build_transformer_config(cfg.model)
    model_type = cfg.model.get("type", "standard_transformer")

    if model_type == "standard_transformer":
        model = StandardTransformerLM(tcfg)

    elif model_type == "recursive_transformer":
        # Use nn-core native recursion support:
        # - set recursive=True
        # - set recurrence_steps=depth
        depth = int(cfg.model.get("depth", 1))
        tcfg.recursive = True
        tcfg.recurrence_steps = depth
        model = StandardTransformerLM(tcfg)

    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    return model, train_loader, val_loader, tok
