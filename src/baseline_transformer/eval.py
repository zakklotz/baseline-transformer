from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _safe_perplexity(val_loss: float) -> float:
    try:
        return math.exp(val_loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def compute_eval_metrics(model, dataloader, device: str, recurrence_steps: int | None = None) -> tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if recurrence_steps is None:
            out = model(**batch)
        else:
            out = model(**batch, recurrence_steps=recurrence_steps)

        logits = out["logits"] if isinstance(out, dict) else out.logits
        labels = batch["labels"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        total_nll += float(nll.item())
        total_tokens += int((shift_labels != -100).sum().item())

    if total_tokens == 0:
        return float("inf"), float("inf")

    val_loss = total_nll / total_tokens
    return val_loss, _safe_perplexity(val_loss)


@torch.no_grad()
def compute_ppl(model, dataloader, device: str, recurrence_steps: int | None = None) -> float:
    _loss, ppl = compute_eval_metrics(model, dataloader, device=device, recurrence_steps=recurrence_steps)
    return ppl
