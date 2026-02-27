from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(self, base_lr: float, warmup_steps: int, max_steps: int, min_lr: float = 0.0):
        self.base_lr = float(base_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_steps = max(1, int(max_steps))
        self.min_lr = float(min_lr)

    def lr_at_step(self, step: int) -> float:
        step = max(0, int(step))

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)

        if step >= self.max_steps:
            return self.min_lr

        decay_steps = max(1, self.max_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def set_optimizer_lr(self, optimizer, step: int) -> float:
        lr = self.lr_at_step(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
