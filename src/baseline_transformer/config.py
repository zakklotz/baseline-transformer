from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExperimentConfig:
    # Minimal “contract” that we keep stable long-term
    model: Dict[str, Any]
    data: Dict[str, Any]
    train: Dict[str, Any]
    seed: int = 1337
    out_dir: str = "runs"

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return ExperimentConfig(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "data": self.data,
            "train": self.train,
            "seed": self.seed,
            "out_dir": self.out_dir,
        }
