import math

import torch

from ecg_chagas_embeddings.callbacks.rolling_mean_metric import RollingMeanMetric


class DummyTrainer:
    def __init__(self):
        self.callback_metrics = {}
        self.world_size = 1


class DummyModule:
    def __init__(self):
        self.logged = []
        self.device = torch.device("cpu")

    def log(self, key, value, **kwargs):
        self.logged.append((key, float(value)))


def test_rolling_mean_metric_default_behavior_uses_available_history():
    trainer = DummyTrainer()
    module = DummyModule()
    cb = RollingMeanMetric(source="val/ap", target="val/ap_mean3", k=3, missing_value=0.0)
    cb.on_fit_start(trainer, module)

    trainer.callback_metrics["val/ap"] = torch.tensor(0.6)
    cb.on_validation_epoch_end(trainer, module)
    assert module.logged[-1] == ("val/ap_mean3", 0.2)


def test_rolling_mean_metric_zero_fills_initial_epochs():
    trainer = DummyTrainer()
    module = DummyModule()
    cb = RollingMeanMetric(
        source="val/ap",
        target="val/ap_mean3",
        k=3,
        missing_value=0.0,
    )
    cb.on_fit_start(trainer, module)

    trainer.callback_metrics["val/ap"] = torch.tensor(0.6)
    cb.on_validation_epoch_end(trainer, module)
    assert module.logged[-1] == ("val/ap_mean3", 0.2)

    trainer.callback_metrics["val/ap"] = torch.tensor(0.3)
    cb.on_validation_epoch_end(trainer, module)
    assert module.logged[-1] == ("val/ap_mean3", 0.3)


def test_rolling_mean_metric_non_finite_source_uses_missing_value():
    trainer = DummyTrainer()
    module = DummyModule()
    cb = RollingMeanMetric(
        source="val/ap",
        target="val/ap_mean3",
        k=3,
        missing_value=0.0,
    )
    cb.on_fit_start(trainer, module)

    trainer.callback_metrics["val/ap"] = torch.tensor(float("nan"))
    cb.on_validation_epoch_end(trainer, module)
    assert module.logged[-1] == ("val/ap_mean3", 0.0)

    trainer.callback_metrics["val/ap"] = torch.tensor(0.6)
    cb.on_validation_epoch_end(trainer, module)
    assert module.logged[-1] == ("val/ap_mean3", 0.2)
    assert not math.isnan(module.logged[-1][1])
