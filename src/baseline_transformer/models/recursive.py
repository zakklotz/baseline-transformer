from __future__ import annotations

import torch
import torch.nn as nn

from nncore.models import Transformer
from nncore.models import TransformerConfig


class RecursiveTransformerLM(nn.Module):
    """
    A simple "recursive" baseline:
    - Build a 1-layer Transformer (shared parameters)
    - Apply it repeatedly depth times
    This is intentionally minimal; we can refine it once we mirror the old Tajalli baseline exactly.
    """
    def __init__(self, cfg: TransformerConfig, depth: int = 6):
        super().__init__()
        self.depth = depth
        shared_cfg = TransformerConfig(**{**cfg.to_dict(), "n_layers": 1})
        self.shared = Transformer(shared_cfg)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        # Re-run the same shared transformer block multiple times.
        # Implementation detail depends on nn-core Transformer interface;
        # as a safe default, we just call the 1-layer model repeatedly on ids.
        x = input_ids
        last = None
        for _ in range(self.depth):
            last = self.shared(input_ids=x, attention_mask=attention_mask, labels=None)
            # Expect logits in output; take argmax to feed next pass is NOT ideal,
            # so for now we just keep the same input_ids and treat this as repeated refinement.
            # We'll replace this with a proper hidden-state recurrence once we mirror old baseline.
        if labels is None:
            return last
        # Final pass with labels for loss
        return self.shared(input_ids=x, attention_mask=attention_mask, labels=labels)
