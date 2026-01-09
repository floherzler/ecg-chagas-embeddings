from __future__ import annotations

import time
from typing import Any, Optional

import torch
from lightning.pytorch.callbacks import Callback


def _infer_batch_size(batch: Any) -> Optional[int]:
    if isinstance(batch, dict):
        for key in ("ecg", "ecg_views", "chagas"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])
        return None
    if isinstance(batch, torch.Tensor) and batch.ndim >= 1:
        return int(batch.shape[0])
    if isinstance(batch, (list, tuple)) and batch:
        return _infer_batch_size(batch[0])
    return None


class ThroughputMonitor(Callback):
    """Logs data/batch timing to help diagnose dataloader stalls."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = int(log_every_n_steps)
        self._last_batch_end_s: Optional[float] = None
        self._batch_start_s: Optional[float] = None
        self._data_times_s: list[float] = []
        self._batch_times_s: list[float] = []
        self._samples_per_s: list[float] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._last_batch_end_s is not None:
            self._data_times_s.append(now - self._last_batch_end_s)
        self._batch_start_s = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._batch_start_s is not None:
            batch_time = now - self._batch_start_s
            self._batch_times_s.append(batch_time)
            batch_size = _infer_batch_size(batch)
            if batch_size is not None and batch_time > 0:
                self._samples_per_s.append(float(batch_size) / float(batch_time))
        self._last_batch_end_s = now

        if self.log_every_n_steps <= 0:
            return
        if trainer.global_step == 0 or (trainer.global_step % self.log_every_n_steps) != 0:
            return

        data_time = (
            float(sum(self._data_times_s) / max(1, len(self._data_times_s)))
            if self._data_times_s
            else 0.0
        )
        batch_time = float(sum(self._batch_times_s) / max(1, len(self._batch_times_s)))
        samples_per_s = (
            float(sum(self._samples_per_s) / max(1, len(self._samples_per_s)))
            if self._samples_per_s
            else 0.0
        )
        wait_pct = 0.0
        denom = data_time + batch_time
        if denom > 0:
            wait_pct = 100.0 * data_time / denom

        pl_module.log(
            "perf/data_time_s",
            data_time,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        pl_module.log(
            "perf/batch_time_s",
            batch_time,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        pl_module.log(
            "perf/samples_per_s",
            samples_per_s,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        pl_module.log(
            "perf/data_wait_pct",
            wait_pct,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

        self._data_times_s.clear()
        self._batch_times_s.clear()
        self._samples_per_s.clear()

