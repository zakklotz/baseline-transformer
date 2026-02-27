from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from nncore.train import Trainer
from nncore.train.optim import build_optimizer  # if exists in nn-core; if not, we’ll inline next commit

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.train.build import build_everything


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.seed)

    model, train_loader, val_loader, _tok = build_everything(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer (simple default, refine later)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train["lr"]),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        grad_clip=float(cfg.train.get("grad_clip", 0.0)),
        log_every=int(cfg.train.get("log_every", 50)),
        eval_every=int(cfg.train.get("eval_every", 500)),
        out_dir=str(cfg.out_dir),
    )
    trainer.train(max_steps=int(cfg.train["max_steps"]))


if __name__ == "__main__":
    main()
