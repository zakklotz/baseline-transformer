from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore.models import TajalliyatConfig
from nncore.models import TajalliyatLM as NNCoreTajalliyatLM


class TajalliyatTransformerLM(nn.Module):
    """
    Adapter around nn-core's TajalliyatLM to support HF-style LM batches:
      input_ids, attention_mask, labels
    """

    def __init__(self, cfg: TajalliyatConfig):
        super().__init__()
        self.model = NNCoreTajalliyatLM(cfg)

    def branch_scheduler_status(
        self,
        *,
        device: torch.device | str,
        compiled: bool | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> dict[str, object]:
        return self.model.branch_scheduler_status(
            device=device,
            compiled=compiled,
            dtype=dtype,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        recurrence_steps: int | None = None,
    ):
        _ = recurrence_steps
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(torch.bool)

        logits = self.model(
            input_ids=input_ids,
            key_padding_mask=key_padding_mask,
        )
        if not torch.is_tensor(logits):
            raise TypeError(f"Unexpected Tajalliyat output type: {type(logits)}")

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return result
