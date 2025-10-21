from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn


def get_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Returns an optimizer configured by name.

    Args:
        name:       Name of optimizer ('sgd', 'adam', 'adamw').
        params:     Iterable of model parameters or parameter groups.
        lr:         Learning rate.
        weight_decay: Weight decay (L2 penalty).
        momentum:   Momentum factor for SGD.
        betas:      Beta coefficients for Adam/AdamW.
        eps:        Epsilon for numerical stability in Adam/AdamW.
        **kwargs:   Additional optimizer-specific keyword args.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs
        )
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, **kwargs)
    elif name in ("adamw", "adam_w", "adam-w"):
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, **kwargs)
    else:
        raise ValueError(
            f"Unsupported optimizer: {name}. Choose one of 'sgd', 'adam', 'adamw'."
        )


def split_optimizer_in_decay_and_no_decay(
    model: nn.Module,
    classifier_decay: float,
    params_decay: float,
    classifier_name: str = "fc",
) -> List[dict[str, Any]]:
    """
    Split model parameters into decay / no-decay groups to control weight decay.
    """
    decay_classifier = []
    decay_other = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name == f"{classifier_name}.weight" and param.ndim > 1:
            decay_classifier.append(param)
        elif param.ndim == 1:
            no_decay.append(param)
        else:
            decay_other.append(param)

    return [
        {"params": decay_classifier, "weight_decay": classifier_decay},
        {"params": decay_other, "weight_decay": params_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
