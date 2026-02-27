import math

import torch

from baseline_transformer.eval import compute_ppl


class DummyLM(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, input_ids, attention_mask=None, labels=None):
        return {"logits": self._logits.to(input_ids.device)}


def test_compute_ppl_uses_token_counts_with_ignore_index():
    logits = torch.tensor(
        [
            [
                [4.0, 1.0],
                [1.0, 4.0],
                [1.0, 4.0],
                [4.0, 1.0],
            ]
        ],
        dtype=torch.float,
    )
    model = DummyLM(logits)

    input_ids = torch.tensor([[0, 1, 1, 0]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    batch_masked = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([[0, 1, -100, 0]], dtype=torch.long),
    }
    ppl_masked = compute_ppl(model, [batch_masked], device="cpu")

    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = batch_masked["labels"][:, 1:].reshape(-1)
    valid = shift_labels != -100
    expected_nll = torch.nn.functional.cross_entropy(
        shift_logits[valid],
        shift_labels[valid],
        reduction="sum",
    )
    expected_ppl = math.exp(float(expected_nll.item()) / int(valid.sum().item()))

    assert math.isclose(ppl_masked, expected_ppl, rel_tol=1e-6)

    batch_unmasked = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([[0, 1, 1, 0]], dtype=torch.long),
    }
    ppl_unmasked = compute_ppl(model, [batch_unmasked], device="cpu")

    assert not math.isclose(ppl_masked, ppl_unmasked, rel_tol=1e-6)
