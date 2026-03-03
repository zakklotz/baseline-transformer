from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.eval import compute_eval_metrics
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


def _tqdm(total: int, desc: str):
    """Optional tqdm progress bar (no hard dependency)."""
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(total=total, desc=desc)
    except Exception:
        return None


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


def _get_curriculum_depth_range(step: int, train_cfg: dict) -> tuple[int, int] | None:
    curriculum = train_cfg.get("depth_curriculum")
    if not isinstance(curriculum, list) or not curriculum:
        return None
    segment = None
    for entry in curriculum:
        end_step = int(entry.get("end_step", 0))
        if step <= end_step:
            segment = entry
            break
    if segment is None:
        segment = curriculum[-1]
    depth_min = segment.get("depth_min")
    depth_max = segment.get("depth_max")
    if depth_min is None or depth_max is None:
        return None
    return int(depth_min), int(depth_max)


def _resolve_recurrence_steps(cfg: ExperimentConfig, optimizer_step: int, recursive: bool) -> int | None:
    if not recursive:
        return None

    default_depth = int(cfg.model.get("depth", 1))
    if not bool(cfg.train.get("variable_depth_training", False)):
        return default_depth

    warmup_steps = int(cfg.train.get("depth_warmup_steps", 0))
    if warmup_steps > 0 and optimizer_step < warmup_steps:
        return int(cfg.train.get("depth_warmup_fixed", default_depth))

    curriculum_range = _get_curriculum_depth_range(optimizer_step, cfg.train)
    if curriculum_range is not None:
        depth_min, depth_max = curriculum_range
        return random.randint(depth_min, depth_max)

    return default_depth


def _load_checkpoint_state(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Checkpoint at {path} is not a training checkpoint with a 'model' entry.")
    return payload


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler_step: int,
    grad_scaler,
    tokens_total: int,
    best_val_ppl: float,
    config: dict,
) -> None:
    stateful_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model": stateful_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_step": scheduler_step,
            "scaler": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
            "tokens_total": tokens_total,
            "best_val_ppl": best_val_ppl,
            "config": config,
            "step": scheduler_step,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Speed knobs (recommended defaults on CUDA) ---
    # These are config-driven so Tajalli can share the same infra later.
    use_tf32 = bool(cfg.train.get("tf32", True)) and device == "cuda"
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # "high" enables TF32 matmul where applicable (Ada GPUs benefit)
        torch.set_float32_matmul_precision("high")

    use_amp = bool(cfg.train.get("amp", True)) and device == "cuda"
    amp_dtype = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if amp_dtype in ("bf16", "bfloat16"):
        autocast_dtype = torch.bfloat16
        need_grad_scaler = False
    elif amp_dtype in ("fp16", "float16"):
        autocast_dtype = torch.float16
        # FP16 usually wants GradScaler; we keep it optional (off by default).
        need_grad_scaler = bool(cfg.train.get("grad_scaler", True))
    else:
        raise ValueError("train.amp_dtype must be one of: bf16, fp16")

    grad_scaler = torch.amp.GradScaler(enabled=(device == "cuda" and use_amp and need_grad_scaler))

    print(
        f"device={device} tf32={use_tf32} amp={use_amp} amp_dtype={amp_dtype} grad_scaler={grad_scaler.is_enabled()}",
        flush=True,
    )

    print("building model + dataloaders...", flush=True)
    model, train_loader, val_loader, _tok = build_everything(cfg)

    base_lr = float(cfg.train["lr"])
    weight_decay = float(cfg.train.get("weight_decay", 0.0))
    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    max_steps = int(cfg.train["max_steps"])
    min_lr = float(cfg.train.get("min_lr", 0.0))
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    grad_accum_steps = max(1, int(cfg.train.get("grad_accum_steps", 1)))
    log_every = int(cfg.train.get("log_every", 50))
    eval_every = int(cfg.train.get("eval_every", 0))
    checkpoint_every = int(cfg.train.get("checkpoint_every", 5000))

    optimizer = build_optimizer(model, lr=base_lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    optimizer_step = 0
    tokens_total = 0
    best_val_ppl = float("inf")
    if args.resume is not None:
        ckpt = _load_checkpoint_state(Path(args.resume))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler_state = ckpt.get("scaler")
        if scaler_state is not None and grad_scaler.is_enabled():
            grad_scaler.load_state_dict(scaler_state)
        optimizer_step = int(ckpt.get("scheduler_step", ckpt.get("step", 0)))
        tokens_total = int(ckpt.get("tokens_total", 0))
        best_val_ppl = float(ckpt.get("best_val_ppl", float("inf")))
        print(f"resumed checkpoint={args.resume} step={optimizer_step} tokens_total={tokens_total}", flush=True)

    scheduler.set_optimizer_lr(optimizer, step=optimizer_step)

    # Compile note: first step(s) can be slower due to compilation overhead.
    model = maybe_compile(model, enabled=bool(args.compile))

    model.to(device)

    seq_len_for_tokens = int(cfg.data.get("block_size", cfg.data.get("max_seq_len", 1)))
    tokens_per_microbatch = int(cfg.train["batch_size"]) * seq_len_for_tokens
    tokens_per_step = tokens_per_microbatch * grad_accum_steps
    model_type = cfg.model.get("type", "standard_transformer")
    recursive = model_type == "recursive_transformer"
    recurrence_steps = int(cfg.model.get("depth", 1)) if recursive else 0
    eval_depth = int(cfg.train.get("eval_depth", 6)) if recursive and bool(cfg.train.get("variable_depth_training", False)) else None

    total_params, trainable_params = count_parameters(model)
    print(f"model.type={model_type} recursive={recursive} recurrence_steps={recurrence_steps}", flush=True)
    print(f"params total={total_params:,} trainable={trainable_params:,}", flush=True)
    print(
        "effective_batch_size="
        f"{int(cfg.train['batch_size']) * grad_accum_steps} "
        f"(batch_size={int(cfg.train['batch_size'])}, grad_accum_steps={grad_accum_steps})",
        flush=True,
    )
    print(f"tokens_per_microbatch~{tokens_per_microbatch}, tokens_per_step~{tokens_per_step}", flush=True)
    print(f"deterministic={deterministic}", flush=True)

    model_stats = save_model_stats(
        out_dir,
        model_type=model_type,
        recursive=recursive,
        recurrence_steps=recurrence_steps,
        total_params=total_params,
        trainable_params=trainable_params,
        effective_batch_size=int(cfg.train["batch_size"]) * grad_accum_steps,
    )
    print(model_stats.strip(), flush=True)

    train_log_path = out_dir / "train_log.csv"
    train_log_exists = train_log_path.exists() and optimizer_step > 0
    train_log_f = train_log_path.open("a" if train_log_exists else "w", encoding="utf-8", newline="")
    train_log = csv.writer(train_log_f)
    if not train_log_exists:
        train_log.writerow(["step", "loss", "lr", "tokens_step", "tokens_total", "wall_time_sec", "depth_used"])

    eval_log_path = out_dir / "eval_log.csv"
    eval_log_exists = eval_log_path.exists() and optimizer_step > 0
    eval_log_f = eval_log_path.open("a" if eval_log_exists else "w", encoding="utf-8", newline="")
    eval_log = csv.writer(eval_log_f)
    if not eval_log_exists:
        eval_log.writerow(["step", "val_perplexity", "val_loss", "eval_depth"])

    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    micro_step = 0
    accum_loss = 0.0
    tokens_this_step = 0
    interval_tokens = 0
    interval_start_time = time.time()
    step_start_time = interval_start_time
    current_recurrence_steps = None

    pbar = _tqdm(max_steps, desc="train steps")
    if pbar is not None and optimizer_step > 0:
        pbar.update(optimizer_step)

    try:
        while optimizer_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            if recursive and micro_step % grad_accum_steps == 0:
                current_recurrence_steps = _resolve_recurrence_steps(cfg, optimizer_step + 1, recursive=True)

            # ---- Forward (AMP if enabled) ----
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = model(**batch, recurrence_steps=current_recurrence_steps)
                    loss = out["loss"] if isinstance(out, dict) else out.loss
            else:
                out = model(**batch, recurrence_steps=current_recurrence_steps)
                loss = out["loss"] if isinstance(out, dict) else out.loss

            loss_value = float(loss.item())
            micro_tokens = _batch_token_count(batch)
            tokens_total += micro_tokens
            tokens_this_step += micro_tokens
            micro_step += 1
            accum_loss += loss_value

            # ---- Backward (GradScaler only if enabled; bf16 doesn't need it) ----
            loss_scaled = loss / grad_accum_steps
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if micro_step % grad_accum_steps != 0:
                continue

            optimizer_step += 1

            if grad_clip > 0:
                if grad_scaler.is_enabled():
                    # Unscale before clipping
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.set_optimizer_lr(optimizer, step=optimizer_step)

            avg_loss = accum_loss / grad_accum_steps
            step_tokens = tokens_this_step
            accum_loss = 0.0
            tokens_this_step = 0

            step_end_time = time.time()
            step_wall_time = step_end_time - step_start_time
            train_log.writerow(
                [
                    optimizer_step,
                    f"{avg_loss:.6f}",
                    f"{lr:.12g}",
                    step_tokens,
                    tokens_total,
                    f"{step_wall_time:.6f}",
                    "" if current_recurrence_steps is None else current_recurrence_steps,
                ]
            )
            train_log_f.flush()

            interval_tokens += step_tokens

            # Update progress bar every optimizer step
            if pbar is not None:
                elapsed = max(step_end_time - interval_start_time, 1e-12)
                tok_s = interval_tokens / elapsed
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", tok_s=f"{tok_s:.0f}")

            if log_every > 0 and optimizer_step % log_every == 0:
                elapsed = max(step_end_time - interval_start_time, 1e-12)
                tokens_per_sec = interval_tokens / elapsed
                print(
                    f"step={optimizer_step}/{max_steps} "
                    f"loss={avg_loss:.4f} lr={lr:.6g} "
                    f"tokens_step={step_tokens} tokens_total={tokens_total} tok_s={tokens_per_sec:.1f} "
                    f"depth={current_recurrence_steps if current_recurrence_steps is not None else '-'}",
                    flush=True,
                )
                interval_start_time = step_end_time
                interval_tokens = 0

            if eval_every > 0 and optimizer_step % eval_every == 0:
                val_loss, ppl = compute_eval_metrics(
                    model,
                    val_loader,
                    device=device,
                    recurrence_steps=eval_depth if recursive else None,
                )
                eval_log.writerow([optimizer_step, f"{ppl:.6f}", f"{val_loss:.6f}", "" if eval_depth is None else eval_depth])
                eval_log_f.flush()
                print(
                    f"eval step={optimizer_step} ppl={ppl:.4f} "
                    f"depth={eval_depth if eval_depth is not None else '-'}",
                    flush=True,
                )
                if ppl < best_val_ppl:
                    best_val_ppl = ppl
                    _save_checkpoint(
                        out_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler_step=optimizer_step,
                        grad_scaler=grad_scaler,
                        tokens_total=tokens_total,
                        best_val_ppl=best_val_ppl,
                        config=cfg.to_dict(),
                    )
                model.train()

            if checkpoint_every > 0 and optimizer_step % checkpoint_every == 0:
                _save_checkpoint(
                    out_dir / f"step_{optimizer_step}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler_step=optimizer_step,
                    grad_scaler=grad_scaler,
                    tokens_total=tokens_total,
                    best_val_ppl=best_val_ppl,
                    config=cfg.to_dict(),
                )
                _save_checkpoint(
                    out_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler_step=optimizer_step,
                    grad_scaler=grad_scaler,
                    tokens_total=tokens_total,
                    best_val_ppl=best_val_ppl,
                    config=cfg.to_dict(),
                )

            step_start_time = time.time()
    finally:
        if pbar is not None:
            pbar.close()
        train_log_f.close()
        eval_log_f.close()

    _save_checkpoint(
        out_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler_step=optimizer_step,
        grad_scaler=grad_scaler,
        tokens_total=tokens_total,
        best_val_ppl=best_val_ppl,
        config=cfg.to_dict(),
    )
    torch.save(getattr(model, "_orig_mod", model).state_dict(), out_dir / "model.pt")
    print(f"saved checkpoint: {out_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
