from __future__ import annotations

import argparse

import torch

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.eval import compute_ppl
from baseline_transformer.train.build import build_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--recurrence-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    model, _train_loader, val_loader, _tok = build_everything(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if args.ckpt is not None:
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)

    ppl = compute_ppl(model, val_loader, device=device, recurrence_steps=args.recurrence_steps)
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
