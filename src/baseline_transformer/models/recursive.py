from __future__ import annotations

from nncore.models import TransformerConfig

from .standard import StandardTransformerLM


class RecursiveTransformerLM(StandardTransformerLM):
    """
    Recursive baseline implemented via nn-core recurrence config flags,
    reusing StandardTransformerLM forward/loss logic.
    """

    def __init__(self, cfg: TransformerConfig, depth: int = 6):
        cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__.copy()
        cfg2 = TransformerConfig.from_dict(cfg_dict)

        cfg2.num_encoder_layers = 0
        cfg2.num_decoder_layers = 1
        cfg2.recursive = True
        cfg2.recurrence_steps = int(depth)

        super().__init__(cfg2)
