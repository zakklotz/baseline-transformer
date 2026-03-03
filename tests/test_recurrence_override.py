import math

import torch

from baseline_transformer.cli.train import _resolve_recurrence_steps
from baseline_transformer.config import ExperimentConfig
from baseline_transformer.eval import compute_eval_metrics
from baseline_transformer.models import RecursiveTransformerLM
from nncore.models import TransformerConfig


def test_recursive_transformer_accepts_recurrence_override():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        max_seq_len=8,
        num_encoder_layers=0,
        num_decoder_layers=1,
    )
    model = RecursiveTransformerLM(cfg, depth=2)
    input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 32, (2, 8), dtype=torch.long)

    default_out = model(input_ids=input_ids, labels=labels)
    override_out = model(input_ids=input_ids, labels=labels, recurrence_steps=4)

    assert default_out["logits"].shape == override_out["logits"].shape
    assert not torch.allclose(default_out["logits"], override_out["logits"])


def test_compute_eval_metrics_supports_recurrence_override():
    cfg = TransformerConfig(
        vocab_size=16,
        d_model=8,
        num_heads=2,
        max_seq_len=6,
        num_encoder_layers=0,
        num_decoder_layers=1,
    )
    model = RecursiveTransformerLM(cfg, depth=2)
    batch = {
        "input_ids": torch.randint(0, 16, (1, 6), dtype=torch.long),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.randint(0, 16, (1, 6), dtype=torch.long),
    }
    loss, ppl = compute_eval_metrics(model, [batch], device="cpu", recurrence_steps=3)
    assert math.isfinite(loss)
    assert math.isfinite(ppl)


def test_resolve_recurrence_steps_uses_curriculum_ranges():
    cfg = ExperimentConfig(
        seed=1337,
        out_dir="runs/test",
        model={"type": "recursive_transformer", "depth": 8, "transformer": {}},
        data={},
        train={
            "variable_depth_training": True,
            "depth_warmup_steps": 10,
            "depth_warmup_fixed": 6,
            "depth_curriculum": [
                {"end_step": 20, "depth_min": 2, "depth_max": 4},
                {"end_step": 30, "depth_min": 5, "depth_max": 7},
            ],
        },
    )

    assert _resolve_recurrence_steps(cfg, optimizer_step=1, recursive=True) == 6
    sampled = _resolve_recurrence_steps(cfg, optimizer_step=25, recursive=True)
    assert sampled is not None
    assert 5 <= sampled <= 7
