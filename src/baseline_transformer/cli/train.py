from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.eval import compute_ppl
from baseline_transformer.train.build import build_everything
from baseline_transformer.train.optim import build_optimizer
from baseline_transformer.train.perf import maybe_compile
from baseline_transformer.train.schedule import WarmupCosineScheduler
from baseline_transformer.utils import (
    count_parameters,
    save_command,
    save_git_commit,
    save_model_stats,
    save_resolved_config,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_token_count(batch: dict[str, torch.Tensor]) -> int:
    labels = batch.get("labels")
    if labels is None:
        return int(batch["input_ids"].numel())
    valid = labels[:, 1:] != -100
    return int(valid.sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.seed)

    deterministic = bool(cfg.train.get("deterministic", False))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg.to_dict(), out_dir)
    save_command(out_dir)
    save_git_commit(out_dir)

    model, train_loader, val_loader, _tok = build_everything(cfg)
    model = maybe_compile(model, enabled=bool(args.compile))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    base_lr = float(cfg.train["lr"])
    weight_decay = float(cfg.train.get("weight_decay", 0.0))
    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    max_steps = int(cfg.train["max_steps"])
    min_lr = float(cfg.train.get("min_lr", 0.0))
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
    log_every = int(cfg.train.get("log_every", 50))
    eval_every = int(cfg.train.get("eval_every", 0))

    optimizer = build_optimizer(model, lr=base_lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )
    scheduler.set_optimizer_lr(optimizer, step=0)


    seq_len_for_tokens = int(cfg.data.get("block_size", cfg.data.get("max_seq_len", 1)))
    tokens_per_microbatch = int(cfg.train["batch_size"]) * seq_len_for_tokens
    tokens_per_step = tokens_per_microbatch * grad_accum_steps
    model_type = cfg.model.get("type", "standard_transformer")
    recursive = model_type == "recursive_transformer"
    recurrence_steps = int(cfg.model.get("depth", 1)) if recursive else 0

    total_params, trainable_params = count_parameters(model)
    print(f"model.type={model_type} recursive={recursive} recurrence_steps={recurrence_steps}")
    print(f"params total={total_params:,} trainable={trainable_params:,}")
    print(
        "effective_batch_size="
        f"{int(cfg.train['batch_size']) * grad_accum_steps} "
        f"(batch_size={int(cfg.train['batch_size'])}, grad_accum_steps={grad_accum_steps})"
    )
    print(f"tokens_per_microbatch~{tokens_per_microbatch}, tokens_per_step~{tokens_per_step}")
    print(f"deterministic={deterministic}")

    model_stats = save_model_stats(
        out_dir,
        model_type=model_type,
        recursive=recursive,
        recurrence_steps=recurrence_steps,
        total_params=total_params,
        trainable_params=trainable_params,
        effective_batch_size=int(cfg.train["batch_size"]) * grad_accum_steps,
    )
    print(model_stats.strip())

    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    micro_step = 0
    optimizer_step = 0
    total_tokens = 0
    accum_loss = 0.0

    while optimizer_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss

        loss_value = float(loss.item())
        total_tokens += _batch_token_count(batch)
        micro_step += 1
        accum_loss += loss_value

        (loss / grad_accum_steps).backward()

        if micro_step % grad_accum_steps != 0:
            continue

        optimizer_step += 1
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr = scheduler.set_optimizer_lr(optimizer, step=optimizer_step)

        avg_loss = accum_loss / grad_accum_steps
        accum_loss = 0.0

        if log_every > 0 and optimizer_step % log_every == 0:
            print(
                f"step={optimizer_step}/{max_steps} "
                f"loss={avg_loss:.4f} lr={lr:.6g} tokens={total_tokens}"
            )

        if eval_every > 0 and optimizer_step % eval_every == 0:
            ppl = compute_ppl(model, val_loader, device=device)
            print(f"eval step={optimizer_step} ppl={ppl:.4f}")
            model.train()

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"saved checkpoint: {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
