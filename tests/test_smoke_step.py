import torch

from baseline_transformer.config import ExperimentConfig
from baseline_transformer.train.build import build_everything


def test_one_forward_step_cpu():
    cfg = ExperimentConfig.load("configs/parity/wt103_512d.yaml")

    model, train_loader, _val_loader, _tok = build_everything(cfg)
    model.to("cpu")
    batch = next(iter(train_loader))
    out = model(**batch)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    assert torch.isfinite(loss).item()
