from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional

import torch
from lightning.pytorch.callbacks import Callback


class RollingMeanMetric(Callback):
    """Log a rolling mean of a metric for smoother monitoring/early stopping."""

    def __init__(self, source: str, target: str, k: int = 3, min_count: int = 1):
        super().__init__()
        if not source:
            raise ValueError("'source' must be a non-empty metric key.")
        if not target:
            raise ValueError("'target' must be a non-empty metric key.")
        if int(k) <= 0:
            raise ValueError("'k' must be > 0.")
        if int(min_count) <= 0:
            raise ValueError("'min_count' must be > 0.")

        self.source = str(source)
        self.target = str(target)
        self.k = int(k)
        self.min_count = int(min_count)
        self._buf: Deque[float] = deque(maxlen=self.k)

    def on_fit_start(self, trainer, pl_module) -> None:
        self._buf.clear()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        current = trainer.callback_metrics.get(self.source)
        current_f: Optional[float]
        if current is None:
            current_f = None
        elif isinstance(current, torch.Tensor):
            current_f = float(current.detach().float().cpu().item())
        else:
            current_f = float(current)

        if current_f is not None and math.isfinite(current_f):
            self._buf.append(current_f)

        mean: float
        if len(self._buf) >= self.min_count:
            mean = float(sum(self._buf) / float(len(self._buf)))
        elif current_f is not None:
            mean = current_f
        else:
            mean = float("nan")

        # Lightning disallows `pl_module.log()` inside certain hooks (e.g. `on_validation_end`).
        # Use the logger directly and also inject into `callback_metrics` so ModelCheckpoint /
        # EarlyStopping can monitor this value.
        if getattr(trainer, "logger", None) is not None:
            try:
                trainer.logger.log_metrics({self.target: mean}, step=trainer.global_step)
            except Exception:
                pass
        try:
            trainer.callback_metrics[self.target] = torch.tensor(
                mean, device=getattr(pl_module, "device", None)
            )
        except Exception:
            pass
