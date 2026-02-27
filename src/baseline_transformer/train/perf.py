from __future__ import annotations

import torch


def maybe_compile(model, enabled: bool):
    if not enabled:
        return model
    if hasattr(torch, "compile"):
        return torch.compile(model)
    return model
