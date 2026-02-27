from __future__ import annotations

import math
import torch


@torch.no_grad()
def compute_ppl(model, dataloader, device: str) -> float:
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss
        losses.append(float(loss.item()))
    mean_loss = sum(losses) / max(1, len(losses))
    return math.exp(mean_loss)
