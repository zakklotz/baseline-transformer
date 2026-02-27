from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    norm_types = (nn.LayerNorm,)
    if hasattr(nn, "RMSNorm"):
        norm_types = norm_types + (nn.RMSNorm,)

    norm_param_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for param_name, _param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                norm_param_names.add(full_name)

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_bias = name.endswith(".bias") or name == "bias"
        is_norm = name in norm_param_names or "norm" in name.lower()
        is_1d = param.ndim < 2

        if is_bias or is_norm or is_1d:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, object]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, lr=lr)
