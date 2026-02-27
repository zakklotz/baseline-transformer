from __future__ import annotations

from typing import Any, Dict

from nncore.models import TransformerConfig


def build_transformer_config(model_cfg: Dict[str, Any]) -> TransformerConfig:
    """
    Map our project config into nn-core's TransformerConfig.
    Expected structure:
      model:
        type: standard_transformer | recursive_transformer
        transformer: {...}   # passed into TransformerConfig(**...)
    """
    if "transformer" not in model_cfg:
        raise KeyError("model.transformer missing from config")

    tcfg = model_cfg["transformer"]
    return TransformerConfig(**tcfg)
