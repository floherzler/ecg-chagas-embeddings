import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth
from tqdm import tqdm
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


from ecg_chagas_embeddings.helper_code import compute_accuracy, compute_challenge_score
from ecg_chagas_embeddings.models.losses import SupConLoss, ConSupPrototypeLoss
from ecg_chagas_embeddings.models.linear_classifier import LinearClassifier
from ecg_chagas_embeddings.utils import (
    get_optimizer,
    split_optimizer_in_decay_and_no_decay,
)
from ecg_chagas_embeddings.metrics.ttc_metrics import (
    calculate_class_alignment_distance,
    calculate_class_alignment_consistency,
    calculate_gaussian_potential_uniformity,
    calculate_sample_alignment_distance,
    calculate_sample_alignment_accuracy,
)


def draw_quantile_bar(
    probs: np.ndarray, width: int = 40, q1char="[", medchar="|", q3char="]"
) -> str:
    if len(probs) == 0:
        return "|<empty>|"

    q1 = np.percentile(probs, 25)
    med = np.median(probs)
    q3 = np.percentile(probs, 75)

    def pos(p):
        return min(width - 1, max(0, int(p * width)))

    bar = ["-"] * width
    bar[pos(q1)] = q1char
    bar[pos(med)] = medchar
    bar[pos(q3)] = q3char

    return "|" + "".join(bar) + "|"


def compute_binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUROC for binary labels without external deps.

    Returns NaN if only one class is present.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    # drop non-finite scores
    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]

    pos = y_true == 1
    neg = y_true == 0
    P = int(pos.sum())
    N = int(neg.sum())
    if P == 0 or N == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = tps / float(P)
    fpr = fps / float(N)

    # add endpoints
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])

    return float(np.trapezoid(tpr, fpr))


def compute_binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Average Precision (AP) for binary labels without external deps.

    This matches sklearn's average_precision_score definition:
    average over precision at each positive when sorting by score descending.
    Returns NaN if no positives are present.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]

    P = int((y_true == 1).sum())
    if P == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = tps / np.maximum(tps + fps, 1)

    return float(precision[y_sorted == 1].sum() / float(P))


def compute_representation_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 2000,
    seed: int = 12345,
):
    """
    Compute per-class TTC embedding metrics (SAD, SAA, CAD, CAC, GPU)
    for embeddings with potentially multiple views per sample.

    Args:
        embeddings: np.ndarray of shape [N, V, D]
            N = number of samples
            V = number of views (augmentations)
            D = embedding dimension
        labels: np.ndarray of shape [N] (0/1 class labels)

    Returns:
        dict: per-class metrics only
    """

    n_samples, n_views, emb_dim = embeddings.shape
    labels = labels.astype(int)

    if max_samples is not None and n_samples > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        n_samples = max_samples
        tqdm.write(f"Subsampled embeddings to {n_samples} samples for TTC metrics.")

    # ---- Flatten to [V*N, D] in view-major order ----
    embs_flat = embeddings.transpose(1, 0, 2).reshape(n_samples * n_views, emb_dim)
    labels_rep = np.tile(labels, n_views)

    # ---- L2 normalize ----
    norms = np.linalg.norm(embs_flat, axis=1, keepdims=True)
    embs_flat = embs_flat / np.clip(norms, a_min=1e-12, a_max=None)

    # ---- Pairwise cosine distances ----
    dot = np.clip(embs_flat @ embs_flat.T, -1.0, 1.0)
    dist_sq = np.clip(2.0 - 2.0 * dot, 0.0, None)
    pairwise_dist = np.sqrt(dist_sq, out=dist_sq)

    metrics = {}
    tqdm.write(
        f"Computing representation metrics for {n_samples} samples, {n_views} views each."
    )

    # ---- SAD / SAA (need >=2 views) ----
    if n_views >= 2:
        sad0, sad1 = calculate_sample_alignment_distance(
            pairwise_dist, n_samples, labels
        )

        # Note: SAA mutates 'sim' by filling its diagonal with inf, so pass a copy
        saa0, saa1 = calculate_sample_alignment_accuracy(
            pairwise_dist.copy(), n_samples, labels, labels_rep
        )

        metrics.update(
            {
                "SAD_0": float(np.mean(sad0)) if sad0.size > 0 else np.nan,
                "SAD_1": float(np.mean(sad1)) if sad1.size > 0 else np.nan,
                "SAA_0": float(saa0),
                "SAA_1": float(saa1),
            }
        )
        tqdm.write(f"Computed metrics for {n_samples} samples, {n_views} views each.")
    else:
        # No second view — metrics undefined
        metrics.update(
            {
                "SAD_0": np.nan,
                "SAD_1": np.nan,
                "SAA_0": np.nan,
                "SAA_1": np.nan,
            }
        )
        tqdm.write(f"Skipped SAD / SAA metrics (only {n_views} view(s) available).")

    # ---- CAD / CAC / GPU (class-wise) ----
    cad0, cad1 = calculate_class_alignment_distance(
        pairwise_dist, embs_flat, labels_rep
    )
    tqdm.write("Computed CAD metrics.")
    cac0, cac1 = calculate_class_alignment_consistency(
        pairwise_dist, embs_flat, labels_rep
    )
    tqdm.write("Computed CAC metrics.")
    gpu0, gpu1 = calculate_gaussian_potential_uniformity(embs_flat, labels_rep)
    tqdm.write("Computed GPU metrics.")

    metrics.update(
        {
            "CAD_0": float(np.mean(cad0)) if cad0.size > 0 else np.nan,
            "CAD_1": float(np.mean(cad1)) if cad1.size > 0 else np.nan,
            "CAC_0": float(np.mean(cac0)) if cac0.size > 0 else np.nan,
            "CAC_1": float(np.mean(cac1)) if cac1.size > 0 else np.nan,
            "GPU_0": float(gpu0),
            "GPU_1": float(gpu1),
        }
    )

    return metrics


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_norm_layer(
    norm_type: str,
    num_channels: int,
    num_groups: int | None = None,
    channels_per_group: int = 8,
    eps: float = 1e-5,
    affine: bool = True,
) -> nn.Module:
    """
    norm_type: one of 'batch', 'instance', 'layer', 'group'
    num_channels: C
    num_groups: only for groupnorm; if None, derived from channels_per_group
    """
    norm_type = norm_type.lower()
    if norm_type == "batch":
        # per-channel over batch + spatial dims
        return nn.BatchNorm1d(num_channels, eps=eps, affine=affine)
    elif norm_type == "instance":
        # per-sample, per-channel over spatial dims
        return nn.InstanceNorm1d(num_channels, eps=eps, affine=affine)
    elif norm_type == "layer":
        # per-sample over all C×spatial dims
        # either use LayerNorm or GroupNorm(1, C)
        return nn.GroupNorm(1, num_channels, eps=eps, affine=affine)
    elif norm_type == "group":
        # per-sample over G groups of channels + spatial dims
        if num_groups is None:
            num_groups = max(1, num_channels // channels_per_group)
        return nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
    elif norm_type in ("none", "identity"):
        # no normalization
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        se_reduction: Optional[int] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        norm_type: str = "group",
        norm_groups: int | None = None,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = norm_layer(norm_type, planes, num_groups=norm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm_layer(norm_type, planes, num_groups=norm_groups)
        self.downsample = downsample  # ty: ignore[unresolved-attribute]
        self.stride = stride  # ty: ignore[unresolved-attribute]
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        ch = planes * self.expansion
        self.se = SELayer1D(ch, se_reduction) if se_reduction else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.stochastic_depth(out)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        se_reduction: Optional[int] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        norm_type: str = "group",
        norm_groups: int | None = None,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm1d
            norm_layer = get_norm_layer
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = norm_layer(norm_type, width, num_groups=norm_groups)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(norm_type, width, num_groups=norm_groups)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = norm_layer(
            norm_type, planes * self.expansion, num_groups=norm_groups
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # ty: ignore[unresolved-attribute]
        self.stride = stride  # ty: ignore[unresolved-attribute]
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        ch = planes * self.expansion
        self.se = SELayer1D(ch, se_reduction) if se_reduction else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.stochastic_depth(out)
        out += identity
        out = self.relu(out)

        return out


class SELayer1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # squeeze: global pooling → (B, C, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # excite: FC → ReLU → FC → Sigmoid
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.size()
        # Squeeze
        y = self.avgpool(x).view(b, c)
        # Excite
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y


BLOCK_REGISTRY = {
    "basic": BasicBlock,
    "bottleneck": Bottleneck,
}


# TODO: fix all the typing issues in this file
class LitResNet18(LightningModule):
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [LitResNet18._serialize_value(v) for v in value]
        return str(value)

    @classmethod
    def _extract_criterion_metadata(cls, criterion: nn.Module) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"class_name": type(criterion).__name__}

        as_config = getattr(criterion, "as_config", None)
        if callable(as_config):
            cfg = as_config()
            if cfg:
                meta["config"] = {
                    key: cls._serialize_value(val) for key, val in cfg.items()
                }

        # fallback: introspect common attributes if no explicit config provided
        if "config" not in meta:
            candidate_attrs = (
                "pos_weight",
                "weight",
                "alpha",
                "gamma",
                "beta",
                "reduction",
            )
            extracted = {}
            for attr in candidate_attrs:
                if hasattr(criterion, attr):
                    extracted[attr] = cls._serialize_value(getattr(criterion, attr))
            if extracted:
                meta["config"] = extracted

        return meta

    def __init__(
        self,
        block: Union[str, Type[Union[BasicBlock, Bottleneck]]] = "basic",
        layers: Sequence[int] = (2, 2, 2, 2),
        inplanes: int = 64,
        num_classes: int = 26,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_type: str = "group",
        norm_groups: int = 4,
        lr: float = 0.0001,
        lr_scheduler: str = "none",
        optimizer: str = "adamw",
        momentum: float = 0.9,
        optimizer_betas: Optional[Sequence[float]] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        classifier_weight_decay: float = 1e-5,
        params_weight_decay: float = 1e-5,
        final_div_factor: float = 1e4,
        one_cycle_pct_start: float = 0.1,
        one_cycle_div_factor: float = 25.0,
        step_size: int = 10,
        step_gamma: float = 0.1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        channels=12,
        initial_kernel_size=7,
        initial_stride=2,
        initial_padding=3,
        stochastic_depth_prob=0.0,
        crop_size=2500,
        max_time_warp=0.15,
        criterion: Optional[nn.Module] = None,
        use_sup_con=False,
        ratio_supervised_majority=0.0,
        use_prototypes=False,
        sup_con_temp=0.07,
        dropout_rate=0.1,
        se_reduction=None,
        track: int = 1,
        pretrained_encoder_path: Optional[str] = None,
        freeze_encoder: bool = False,
        unfreeze_last_block: bool = False,
        log_umap: bool = True,
        umap_n_neighbors: int = 50,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        umap_n_epochs: int = 250,
        umap_seed: Optional[int] = None,
        umap_log_every_n_epochs: int = 1,
        init_classifier_bias: bool = False,
        classifier_bias_pos_fraction: Optional[float] = None,
        use_linear_probe_head: bool = False,
    ) -> None:
        super().__init__()

        if optimizer_betas is not None:
            optimizer_betas = tuple(optimizer_betas)

        if isinstance(block, str):
            block_key = block.lower()
            if block_key not in BLOCK_REGISTRY:
                raise ValueError(
                    f"Unknown block '{block}'. Available options: {list(BLOCK_REGISTRY)}"
                )
            block_cls = BLOCK_REGISTRY[block_key]
        else:
            block_cls = block

        layers = tuple(layers)
        if len(layers) != 4:
            raise ValueError(
                f"Expected 'layers' to have 4 elements (got {len(layers)}). "
                "Adjust the architecture helpers if you require a different depth."
            )

        self.block_type = block_cls  # ty: ignore[unresolved-attribute]
        self.layers_config = layers  # ty: ignore[unresolved-attribute]

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = list(replace_stride_with_dilation)
            if len(replace_stride_with_dilation) != 3:
                raise ValueError(
                    "replace_stride_with_dilation should be None "
                    f"or a 3-element tuple, got {replace_stride_with_dilation}"
                )

        if norm_layer is None:
            norm_layer = get_norm_layer
        self._norm_layer = norm_layer  # ty: ignore[unresolved-attribute]
        self.norm_groups = norm_groups  # ty: ignore[unresolved-attribute]
        self.max_lr = lr  # ty: ignore[unresolved-attribute]
        self.lr_scheduler = lr_scheduler  # ty: ignore[unresolved-attribute]
        self.final_div_factor = final_div_factor  # ty: ignore[unresolved-attribute]
        self.one_cycle_pct_start = one_cycle_pct_start  # ty: ignore[unresolved-attribute]
        self.one_cycle_div_factor = one_cycle_div_factor  # ty: ignore[unresolved-attribute]
        self.step_size = step_size  # ty: ignore[unresolved-attribute]
        self.step_gamma = step_gamma  # ty: ignore[unresolved-attribute]

        self.inplanes = inplanes  # ty: ignore[unresolved-attribute]
        self.dilation = 1  # ty: ignore[unresolved-attribute]
        self.channels = channels  # ty: ignore[unresolved-attribute]
        self.initial_kernel_size = initial_kernel_size  # ty: ignore[unresolved-attribute]
        self.initial_stride = initial_stride  # ty: ignore[unresolved-attribute]
        self.initial_padding = initial_padding  # ty: ignore[unresolved-attribute]
        self.stochastic_depth_prob = stochastic_depth_prob  # ty: ignore[unresolved-attribute]
        self.crop_size = crop_size  # ty: ignore[unresolved-attribute]
        self.max_time_warp = max_time_warp  # ty: ignore[unresolved-attribute]
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.se_reduction = se_reduction  # ty: ignore[unresolved-attribute]
        self.track = int(track)  # ty: ignore[unresolved-attribute]
        if self.track not in (1, 2, 3):
            raise ValueError(f"Unsupported track={self.track}. Use 1, 2, or 3.")
        if self.track == 1 and (use_sup_con or use_prototypes):
            raise ValueError("Track 1 is classifier-only; disable SupCon/Prototypes.")
        if self.track == 2 and not (use_sup_con or use_prototypes):
            raise ValueError("Track 2 requires SupCon or Prototypes enabled.")
        if self.track == 3 and (use_sup_con or use_prototypes):
            raise ValueError("Track 3 is classifier-only; disable SupCon/Prototypes.")

        self.use_sup_con = use_sup_con
        self.ratio_supervised_majority = ratio_supervised_majority
        self.use_prototypes = use_prototypes
        self.use_classifier = self.track in (1, 3)
        self.sup_con_temp = sup_con_temp  # ty: ignore[unresolved-attribute]
        self.pretrained_encoder_path = pretrained_encoder_path  # ty: ignore[unresolved-attribute]
        self.freeze_encoder = freeze_encoder  # ty: ignore[unresolved-attribute]
        self.unfreeze_last_block = unfreeze_last_block  # ty: ignore[unresolved-attribute]
        self._encoder_initialized = False  # ty: ignore[unresolved-attribute]
        self.use_linear_probe_head = bool(use_linear_probe_head)  # ty: ignore[unresolved-attribute]
        if self.use_linear_probe_head and self.track != 3:
            raise ValueError("`use_linear_probe_head` is intended for track 3 only.")

        self.log_umap = log_umap  # ty: ignore[unresolved-attribute]
        self.umap_n_neighbors = umap_n_neighbors  # ty: ignore[unresolved-attribute]
        self.umap_min_dist = umap_min_dist  # ty: ignore[unresolved-attribute]
        self.umap_metric = umap_metric  # ty: ignore[unresolved-attribute]
        self.umap_n_epochs = umap_n_epochs  # ty: ignore[unresolved-attribute]
        self.umap_seed = umap_seed  # ty: ignore[unresolved-attribute]
        self.umap_log_every_n_epochs = int(max(1, umap_log_every_n_epochs))  # ty: ignore[unresolved-attribute]
        self._umap_neg_ids: Optional[List[str]] = None  # ty: ignore[unresolved-attribute]

        self.init_classifier_bias = init_classifier_bias  # ty: ignore[unresolved-attribute]
        self.classifier_bias_pos_fraction = classifier_bias_pos_fraction  # ty: ignore[unresolved-attribute]
        self._classifier_bias_initialized = False  # ty: ignore[unresolved-attribute]
        self.fake_sup_dropout = nn.Dropout(p=0.1)
        self.fake_sup_noise_std = 0.01  # ty: ignore[unresolved-attribute]
        self.sup_con_loss = SupConLoss(
            temperature=self.sup_con_temp,
            contrast_mode="ALL_VIEWS",
            base_temperature=self.sup_con_temp,
            ratio_supervised_majority=self.ratio_supervised_majority,
            min_class=1,
        )
        self.proto_loss = ConSupPrototypeLoss(
            temperature=self.sup_con_temp,
            base_temperature=self.sup_con_temp,
            minority_cls=1,  # 1 = Chagas as minority in your setup
            eps=0.1,  # start default; tune later
            eps_0=0.1,
            eps_1=0.1,
            negatives_weight=1.0,
        )
        # keep string-friendly block identifier in the saved hyperparameters
        block_hparam = block if isinstance(block, str) else block_cls.__name__
        criterion_meta = self._extract_criterion_metadata(self.criterion)
        self.save_hyperparameters(ignore=["criterion"])
        self.hparams.update({"block": block_hparam, "criterion_meta": criterion_meta})
        if optimizer_betas is not None:
            # make sure betas are logged as a regular list for readability
            self.hparams.optimizer_betas = list(optimizer_betas)  # ty: ignore[unresolved-attribute]
        self.hparams.optimizer_eps = optimizer_eps  # ty: ignore[unresolved-attribute]
        self.hparams.layers = list(self.layers_config)  # ty: ignore[unresolved-attribute]
        self.hparams.replace_stride_with_dilation = list(replace_stride_with_dilation)  # ty: ignore[unresolved-attribute]

        self.train_step_losses = []  # ty: ignore[unresolved-attribute]
        self.train_step_supcon_losses = []  # ty: ignore[unresolved-attribute]
        self.val_step_losses = []  # ty: ignore[unresolved-attribute]
        self.validation_step_outputs = []  # ty: ignore[unresolved-attribute]
        self._pred_rows = []  # ty: ignore[unresolved-attribute]

        self.groups = groups  # ty: ignore[unresolved-attribute]
        self.base_width = width_per_group  # ty: ignore[unresolved-attribute]
        self.conv1 = nn.Conv1d(
            channels,
            self.inplanes,
            kernel_size=self.initial_kernel_size,
            stride=self.initial_stride,
            padding=self.initial_padding,
            bias=False,
        )
        self.norm1 = norm_layer(norm_type, self.inplanes, num_groups=self.norm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self._stage_block_id = 0  # ty: ignore[unresolved-attribute]
        self._sd_prob = 0.0  # ty: ignore[unresolved-attribute]
        self._total_layers = sum(layers)  # ty: ignore[unresolved-attribute]

        self.layer1 = self._make_layer(
            block_cls,
            norm_type,
            norm_groups,
            inplanes,
            layers[0],
            se_reduction=self.se_reduction,
        )
        self.layer2 = self._make_layer(
            block_cls,
            norm_type,
            norm_groups,
            inplanes * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            se_reduction=self.se_reduction,
        )
        self.layer3 = self._make_layer(
            block_cls,
            norm_type,
            norm_groups,
            inplanes * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            se_reduction=self.se_reduction,
        )
        self.layer4 = self._make_layer(
            block_cls,
            norm_type,
            norm_groups,
            inplanes * 8,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            se_reduction=self.se_reduction,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.fc = nn.Linear(inplanes * 8 * block_cls.expansion, num_classes)

        feat_dim = inplanes * 8 * block_cls.expansion
        self.linear_probe_head = (
            LinearClassifier(
                input_size=feat_dim,
                num_classes=num_classes,
                p_dropout=dropout_rate,
            )
            if self.use_linear_probe_head
            else None
        )

        if use_sup_con or use_prototypes:
            self.projection_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
            )
        else:
            self.projection_head = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.norm3.weight is not None:
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.norm2.weight is not None:
                    nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        norm_type: str,
        norm_groups: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        se_reduction=None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(norm_type, planes * block.expansion, num_groups=norm_groups),
            )

        layers = []

        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                se_reduction,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                norm_type=norm_type,
                norm_groups=norm_groups,
                stochastic_depth_prob=self._get_and_update_stochastic_depth_prob(),
            )
        )
        self.inplanes = planes * block.expansion  # ty: ignore[unresolved-attribute]
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    se_reduction=se_reduction,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    stochastic_depth_prob=self._get_and_update_stochastic_depth_prob(),
                )
            )

        return nn.Sequential(*layers)

    def _get_and_update_stochastic_depth_prob(self):
        sd_prob = (
            self.stochastic_depth_prob
            * self._stage_block_id
            / (self._total_layers - 1.0)
        )
        self._stage_block_id += 1
        return sd_prob

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        feats = self.dropout(feats)
        if self.use_linear_probe_head and self.linear_probe_head is not None:
            logits = self.linear_probe_head(feats)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
        else:
            logits = self.fc(feats)

        return feats, logits

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feats, logits = self._forward_impl(x)
        proj = self.projection_head(feats)
        return feats, proj, logits

    def setup(self, stage: Optional[str] = None) -> None:
        if self._encoder_initialized:
            return

        if self.track == 3:
            if not self.pretrained_encoder_path:
                raise ValueError(
                    "Track 3 requires 'pretrained_encoder_path' to load encoder weights."
                )
            self._load_pretrained_encoder(self.pretrained_encoder_path)

        self._apply_freeze_policy()
        self._encoder_initialized = True

    def _encoder_module_prefixes(self) -> Tuple[str, ...]:
        return ("conv1", "norm1", "layer1", "layer2", "layer3", "layer4", "avgpool")

    def _load_pretrained_encoder(self, path: str) -> None:
        resolved_path = self._resolve_wandb_artifact(path)
        ckpt = torch.load(resolved_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        prefixes = self._encoder_module_prefixes()
        for key, value in state_dict.items():
            k = key
            if k.startswith("model."):
                k = k[len("model.") :]
            if k.startswith(prefixes):
                cleaned[k] = value
        incompatible = self.load_state_dict(cleaned, strict=False)
        if incompatible.missing_keys:
            tqdm.write(
                f"Missing keys when loading encoder: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            tqdm.write(
                f"Unexpected keys when loading encoder: {incompatible.unexpected_keys}"
            )

    def _resolve_wandb_artifact(self, path: str) -> str:
        """
        Resolve a wandb artifact URI to a local file path. If `path` does not start with
        'wandb:', it is returned unchanged.

        Supported format: wandb:<entity>/<project>/<artifact-name>:<alias>
        Example: wandb:ag-lukassen/ecg-chagas-embeddings-cli/model-51nqcxar:v1
        """
        if not path.startswith("wandb:"):
            return path

        uri = path[len("wandb:") :]
        api = wandb.Api()
        art = api.artifact(uri, type="model")
        local_dir = Path(art.download())

        # Prefer common Lightning checkpoint names
        for fname in ("model.ckpt", "best.ckpt"):
            candidate = local_dir / fname
            if candidate.exists():
                return str(candidate)

        # Fallback: first .ckpt in the artifact
        ckpts = list(local_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found in wandb artifact {uri}")
        return str(ckpts[0])

    def _apply_freeze_policy(self) -> None:
        if not self.use_classifier:
            for param in self.fc.parameters():
                param.requires_grad = False

        if not self.freeze_encoder:
            return

        for name, param in self.named_parameters():
            if name.startswith(self._encoder_module_prefixes()):
                param.requires_grad = False

        # Keep encoder BatchNorms frozen as well so running stats do not drift
        # when the encoder is meant to stay fixed.
        for name, module in self.named_modules():
            if not name.startswith(self._encoder_module_prefixes()):
                continue
            if self.unfreeze_last_block and name.startswith("layer4"):
                continue
            if isinstance(module, nn.BatchNorm1d):
                module.eval()  # keep running stats fixed

        if self.unfreeze_last_block:
            for param in self.layer4.parameters():
                param.requires_grad = True

    def on_fit_start(self):
        # Ensure nested loss modules pick up the correct device.
        # They are implemented as LightningModules and rely on their internal `.device`
        # property when creating tensors.
        if self.use_sup_con:
            self.sup_con_loss = self.sup_con_loss.to(self.device)
        if self.use_prototypes:
            self.proto_loss = self.proto_loss.to(self.device)

        if not self.use_prototypes:
            self._maybe_init_classifier_bias()
            return
        with torch.no_grad():
            # If projection_head is a Sequential with a final Linear, use its out_features,
            # otherwise fall back to the feature dimensionality stored in fc.in_features
            D = None
            if (
                isinstance(self.projection_head, nn.Sequential)
                and len(self.projection_head) > 0
            ):
                last_module = self.projection_head[-1]
                if hasattr(last_module, "out_features"):
                    D = int(last_module.out_features)
            if D is None:
                # fallback to feature dimension produced before the classifier
                if hasattr(self.fc, "in_features"):
                    D = int(self.fc.in_features)
                else:
                    # ultimate fallback: run a dummy forward pass to infer dimension
                    dummy = torch.zeros(
                        1, getattr(self, "inplanes", 1), device=self.device
                    )
                    # try to get features size by running through _forward_impl -> projection_head
                    feats, _ = self._forward_impl(dummy)
                    proj = self.projection_head(feats)
                    D = int(proj.shape[-1])

            u = torch.randn(D, device=self.device)
            u = u / (u.norm() + 1e-6)
            prototypes = torch.stack([u, -u], dim=0)  # [2, D]
        self.proto_loss.set_prototypes(prototypes)  # required
        self._maybe_init_classifier_bias()

    def _maybe_init_classifier_bias(self) -> None:
        if not self.init_classifier_bias or self._classifier_bias_initialized:
            return
        if not self.use_classifier:
            return
        if getattr(self, "trainer", None) is not None and getattr(
            self.trainer, "ckpt_path", None
        ):
            # Avoid mutating restored weights when resuming from a checkpoint.
            return
        if not isinstance(self.fc, nn.Linear) or self.fc.bias is None:
            return
        if getattr(self.fc, "out_features", 1) != 1:
            tqdm.write(
                "Skipping classifier bias init (only supported for binary num_classes=1)."
            )
            return

        pos_fraction = self.classifier_bias_pos_fraction
        if pos_fraction is None and getattr(self, "trainer", None) is not None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                pos_fraction = getattr(dm, "pos_fraction", None)
                if (
                    pos_fraction is None
                    and hasattr(dm, "cfg")
                    and isinstance(dm.cfg, dict)
                ):
                    pos_fraction = dm.cfg.get("pos_fraction", None)

        if pos_fraction is None:
            tqdm.write(
                "Skipping classifier bias init (pos_fraction unavailable). "
                "Set `model.classifier_bias_pos_fraction` or provide `data.pos_fraction`."
            )
            return

        p = float(pos_fraction)
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        bias = float(math.log(p / (1.0 - p)))
        with torch.no_grad():
            self.fc.bias.fill_(bias)
        self._classifier_bias_initialized = True
        tqdm.write(f"Initialized classifier bias to logit(p)={bias:.4f} for p={p:.6f}")

    def training_step(self, batch, batch_idx):
        labels = batch["chagas"].view(-1)  # [B]
        y_float = labels.to(torch.float32)  # for focal/BCE-like classifier loss
        y_long = labels.to(torch.long)  # for one_hot / supcon
        ages = batch.get("age", None)
        sexes = batch.get("sex", None)
        sources = batch.get("source", None)
        ids = batch.get("exam_id", None)

        def compute_cls_loss(logits):
            # logits: [B] or [B,1] or [B,num_classes]
            metadata = {
                "labels": labels,
                "source": sources,
                "exam_id": ids,
                "age": ages,
                "sex": sexes,
            }
            crit_name = type(self.criterion).__name__
            if (
                crit_name == "SourceWeightedBCE"
                or crit_name == "SourceWeightedTopTverskyLoss"
            ):
                return self.criterion(logits, labels, metadata)
            else:
                return self.criterion(logits, labels)

        # ------ SupCon path ------
        if self.use_sup_con:
            if "ecg_views" not in batch:
                raise ValueError(
                    "use_sup_con=True requires 'ecg_views' in batch. "
                    "Check dataloader n_views or transforms."
                )
            x = batch["ecg_views"]  # [B,2,C,T]
            if x.dim() != 4 or x.shape[1] < 2:
                raise ValueError(
                    f"'ecg_views' must be [B,V,C,T] with V>=2 for SupCon; "
                    f"got shape {tuple(x.shape)}."
                )
            B, V, C, T = x.shape
            x = x.view(B * V, C, T)

            feats, proj, logits = self(x)
            proj = F.normalize(proj, dim=1, eps=1e-6).view(B, V, -1)
            logits = logits.view(B, V, -1).mean(dim=1).squeeze(-1)  # [B]

            if self.use_classifier:
                # classifier wants FLOAT targets
                cls_loss = self.criterion(logits, y_float)
            else:
                cls_loss = torch.tensor(0.0, device=logits.device)

            # SupCon can use INT labels
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                con_loss, *_ = self.sup_con_loss(proj.float(), y_long)

            if self.use_classifier:
                self.train_step_losses.append(cls_loss.detach())
            self.train_step_supcon_losses.append(con_loss.detach())

            return cls_loss + con_loss

        elif self.use_prototypes:
            if "ecg_views" not in batch:
                raise ValueError(
                    "use_prototypes=True requires 'ecg_views' in batch. "
                    "Check dataloader n_views or transforms."
                )
            x = batch["ecg_views"]  # [B,2,C,T]
            if x.dim() != 4 or x.shape[1] < 2:
                raise ValueError(
                    f"'ecg_views' must be [B,V,C,T] with V>=2 for prototypes; "
                    f"got shape {tuple(x.shape)}."
                )
            B, V, C, T = x.shape
            x = x.view(B * V, C, T)

            feats, proj, logits = self(x)
            proj = F.normalize(proj, dim=1, eps=1e-6).view(B, V, -1)
            logits = logits.view(B, V, -1).mean(dim=1).squeeze(-1)  # [B]

            if self.use_classifier:
                # classifier wants FLOAT targets
                cls_loss = self.criterion(logits, y_float)
            else:
                cls_loss = torch.tensor(0.0, device=logits.device)

            # prototype loss wants ONE-HOT (FLOAT)
            y_oh = F.one_hot(y_long, num_classes=2).to(torch.float32)
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                proto_loss, *_ = self.proto_loss(proj.float(), y_oh)

            if self.use_classifier:
                self.train_step_losses.append(cls_loss.detach())
            self.train_step_supcon_losses.append(proto_loss.detach())

            return cls_loss + proto_loss

        # ------ Classification-only path ------
        else:
            x = batch.get("ecg")
            if x is None and "ecg_views" in batch:
                x = batch["ecg_views"][:, 0]  # fall back to first view
            if x is None:
                raise ValueError(
                    "Classification-only path requires 'ecg' or 'ecg_views' in batch."
                )
            feats, proj, logits = self(x)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)  # [B]

            cls_loss = compute_cls_loss(logits)
            self.train_step_losses.append(cls_loss.detach())
            return cls_loss

    def validation_step(self, batch, batch_idx):
        labels = batch["chagas"].view(-1)
        ages = batch.get("age", None)
        sexes = batch.get("sex", None)
        sources = batch.get("source", None)
        ids = batch.get("exam_id", None)
        needs_views = self.use_sup_con or self.use_prototypes
        has_views = "ecg_views" in batch
        if needs_views and not has_views:
            raise ValueError(
                "use_sup_con/use_prototypes=True requires 'ecg_views' in batch. "
                "Check dataloader n_views or transforms."
            )

        if has_views:
            views = batch["ecg_views"]  # [B, V, C, T]
            if views.dim() != 4:
                raise ValueError(
                    f"'ecg_views' must be [B,V,C,T]; got shape {tuple(views.shape)}."
                )
            if needs_views and views.shape[1] < 2:
                raise ValueError(
                    f"'ecg_views' must have V>=2 for SupCon/Prototypes; "
                    f"got shape {tuple(views.shape)}."
                )
            B, V, C, T = views.shape
            flat = views.view(B * V, C, T)
            feats, proj, logits = self(flat)
            proj = F.normalize(proj, dim=1, eps=1e-6).view(B, V, -1)
            logits = logits.view(B, V, -1)
            if needs_views:
                logits = logits.mean(dim=1).squeeze(-1)
            else:
                logits = logits[:, 0].squeeze(-1)
        else:
            signals = batch.get("ecg")
            if signals is None:
                raise ValueError("Validation requires 'ecg' or 'ecg_views' in batch.")
            feats, proj, logits = self(signals)
            proj = F.normalize(proj, dim=1, eps=1e-6).unsqueeze(1)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

        # print(f"Sources: {sources}")

        metadata = {
            "labels": labels,
            "source": sources,
            "exam_id": ids,
            "age": ages,
            "sex": sexes,
        }

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # [B]
        probs = torch.sigmoid(logits)
        if torch.isnan(probs).any():
            print("NaN in probs")
            print(f"probs: {probs}")
            print(f"logits: {logits}")
            print(f"labels: {labels}")
            print(f"exam_ids: {ids}")
        preds = (probs > 0.5).long()
        if self.use_classifier:
            if type(self.criterion).__name__ == "SourceWeightedBCE":
                loss = self.criterion(logits, labels, metadata)
            elif type(self.criterion).__name__ == "SourceWeightedTopTverskyLoss":
                loss = self.criterion(logits, labels, metadata)
            else:
                loss = self.criterion(logits, labels)
        else:
            loss = torch.tensor(0.0, device=logits.device)
        self.validation_step_outputs.append(
            (
                labels.view(-1),
                probs.view(-1),
                preds.view(-1),
                ages.view(-1) if ages is not None else None,
                sexes.view(-1) if sexes is not None else None,
                sources.view(-1) if sources is not None else None,
                ids if ids is not None else None,
                proj.detach().to(device="cpu", dtype=torch.float32),  # [B,V,D]
            )
        )
        self.val_step_losses.append(loss)
        return {"val_loss": loss, "gt": labels, "probs": probs, "preds": preds}

    def on_validation_epoch_end(self):
        gts = (
            torch.cat([b[0] for b in self.validation_step_outputs], dim=0).cpu().numpy()
        )
        probs = (
            torch.cat([b[1] for b in self.validation_step_outputs], dim=0)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        preds = (
            torch.cat([b[2] for b in self.validation_step_outputs], dim=0).cpu().numpy()
        )
        # ages = (
        #     torch.cat([b[3] for b in self.validation_step_outputs], dim=0).cpu().numpy()
        #     if self.validation_step_outputs[0][3] is not None
        #     else None
        # )
        # sexes = (
        #     torch.cat([b[4] for b in self.validation_step_outputs], dim=0).cpu().numpy()
        #     if self.validation_step_outputs[0][4] is not None
        #     else None
        # )
        sources = (
            torch.cat([b[5] for b in self.validation_step_outputs], dim=0).cpu().numpy()
            if self.validation_step_outputs[0][5] is not None
            else None
        )
        ids: Optional[List[str]] = None
        if (
            self.validation_step_outputs
            and self.validation_step_outputs[0][6] is not None
        ):
            ids = []
            for batch_out in self.validation_step_outputs:
                batch_ids = batch_out[6]
                if batch_ids is None:
                    continue
                if isinstance(batch_ids, (list, tuple)):
                    ids.extend([str(x) for x in batch_ids])
                elif isinstance(batch_ids, torch.Tensor):
                    ids.extend([str(x) for x in batch_ids.detach().cpu().tolist()])
                else:
                    ids.append(str(batch_ids))
        embeddings = (
            torch.cat([b[7] for b in self.validation_step_outputs], dim=0).cpu().numpy()
        )
        tqdm.write(f"Embeddings shape: {embeddings.shape}")

        # if sources is not None:
        #    unique_sources, counts = np.unique(sources, return_counts=True)
        #    tqdm.write("Source counts:")
        #    for src, count in zip(unique_sources, counts):
        #        tqdm.write(f"  Source {src}: {count}")

        self.validation_step_outputs.clear()

        # num_gts_ones = np.sum(gts == 1.0)
        # tqdm.write(f"Positive cases in epoch: {num_gts_ones} of total {len(gts)}")

        acc = compute_accuracy(gts, preds)
        score = 0.0
        try:
            score = compute_challenge_score(gts, probs)
            code15_gts = gts[sources == 0] if sources is not None else gts
            code15_probs = probs[sources == 0] if sources is not None else probs
            ptb_xl_gts = gts[sources == 1] if sources is not None else gts
            # ptb_xl_probs = probs[sources == 1] if sources is not None else probs
            sami_trop_gts = gts[sources == 2] if sources is not None else gts
            # sami_trop_probs = probs[sources == 2] if sources is not None else probs
            code15_score = None
            code15_accuracy = None
            strong_score = None
            strong_accuracy = None
            if code15_gts.size > 0 or code15_probs.size > 0:
                tqdm.write("CODE-15 confusion matrix:")
                code15_score = (
                    compute_challenge_score(gts[sources == 0], probs[sources == 0])
                    if sources is not None
                    else None
                )
                code15_accuracy = (
                    compute_accuracy(gts[sources == 0], preds[sources == 0])
                    if sources is not None
                    else None
                )
                self.log(
                    "val/code15_acc",
                    code15_accuracy,
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                )
                self.log(
                    "val/code15_score",
                    code15_score,
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                )
                tqdm.write(
                    f"CODE-15 score: {code15_score:.4f}, accuracy: {code15_accuracy:.4f}"
                )
            if sami_trop_gts.size > 0 and ptb_xl_gts.size > 0:
                tqdm.write("Strong Labels confusion matrix:")
                strong_score = (
                    compute_challenge_score(gts[sources != 0], probs[sources != 0])
                    if sources is not None
                    else None
                )
                strong_accuracy = (
                    compute_accuracy(gts[sources != 0], preds[sources != 0])
                    if sources is not None
                    else None
                )
                tqdm.write(
                    f"Strong Labels score: {strong_score:.4f}, accuracy: {strong_accuracy:.4f}"
                )
                self.log(
                    "val/strong_score",
                    strong_score,
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                )
                self.log(
                    "val/strong_acc",
                    strong_accuracy,
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                )
        except Exception as e:
            tqdm.write(
                f"Error in computing challenge score: {repr(e)}. Setting score to 0.0"
            )

        # Build a balanced subset (all positives + same number of negatives) for metrics/UMAP.
        subset_idx = self._select_balanced_subset_indices(gts, ids)
        if subset_idx.size == 0:
            tqdm.write("Skipping embedding metrics/UMAP logging (no balanced subset).")
            emb_metrics = {}
            emb_subset = embeddings
            gts_subset = gts
            sources_subset = sources
            ids_subset = ids or []
        else:
            emb_subset = embeddings[subset_idx]
            gts_subset = gts[subset_idx]
            sources_subset = sources[subset_idx] if sources is not None else None
            ids_subset = [ids[i] for i in subset_idx] if ids is not None else None
        try:
            emb_metrics = compute_representation_metrics(
                emb_subset,
                gts_subset,
                max_samples=emb_subset.shape[0],
                seed=int(self.umap_seed or 12345),
            )
            for k, v in emb_metrics.items():
                tqdm.write(f"Embedding metric {k}: {v:.4f}")
                self.log(f"emb_{k}", v, prog_bar=False, on_epoch=True, on_step=False)
        except Exception as e:
            emb_metrics = {}
            tqdm.write(f"Error in computing representation metrics: {repr(e)}")

        try:
            if self.log_umap and (
                self.current_epoch % self.umap_log_every_n_epochs == 0
            ):
                self._log_umap_diagnostics(
                    emb_subset,
                    gts_subset,
                    ids=ids_subset,
                    sources=sources_subset,
                    precomputed_metrics=emb_metrics,
                )
        except Exception as e:
            tqdm.write(f"Error in computing/logging UMAP diagnostics: {repr(e)}")

        quartiles = {}

        # tqdm.write(f"probs: {probs[:5]}")
        tqdm.write(f"... Quartiles for score {score:.2f}:")
        for src in ["strong", "weak", "all"]:
            if src == "strong":
                src_probs = probs[sources != 0] if sources is not None else probs
                src_gts = gts[sources != 0] if sources is not None else gts
            elif src == "weak":
                src_probs = probs[sources == 0] if sources is not None else probs
                src_gts = gts[sources == 0] if sources is not None else gts
            else:
                src_probs = probs
                src_gts = gts
            for cls in [0, 1]:
                cls_probs = src_probs[src_gts == cls]
                if len(cls_probs) == 0:
                    quartiles[f"{src}_class_{cls}"] = {
                        "q1": 0.0,
                        "median": 0.0,
                        "q3": 0.0,
                    }
                    q1 = med = q3 = 0.0
                else:
                    cls_probs = np.asarray(cls_probs, dtype=np.float64)
                    q1 = float(np.nanpercentile(cls_probs, 25))
                    med = float(np.nanmedian(cls_probs))
                    q3 = float(np.nanpercentile(cls_probs, 75))
                    quartiles[f"{src}_class_{cls}"] = {
                        "q1": q1,
                        "median": med,
                        "q3": q3,
                    }
                # tqdm.write(f"    {src} class {cls} -> q1: {q1:.4f}, median: {med:.4f}, p95: {q3:.4f}")
                bar = draw_quantile_bar(
                    cls_probs, width=50, q1char="◁", medchar="●", q3char="▷"
                )
                tqdm.write(
                    f"{src} class {cls}:\t {bar} -> q1: {q1:.2f}, median: {med:.2f}, q3: {q3:.2f}"
                )

        # Log quartiles
        for cls, stats in quartiles.items():
            for k, v in stats.items():
                self.log(
                    f"val/{cls}_{k}", v, on_epoch=True, on_step=False, prog_bar=False
                )

        self.log("val/acc", acc, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val/score", score, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_score", score, prog_bar=False, on_epoch=True, on_step=False)

        # Standard, more interpretable metrics alongside challenge score.
        auroc = compute_binary_auroc(gts, probs)
        ap = compute_binary_average_precision(gts, probs)
        self.log("val/auroc", auroc, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val/ap", ap, prog_bar=False, on_epoch=True, on_step=False)

        if sources is not None:
            # Per-source subsets (may be NaN if a subset has a single class)
            self.log(
                "val/code15_auroc",
                compute_binary_auroc(gts[sources == 0], probs[sources == 0]),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "val/code15_ap",
                compute_binary_average_precision(
                    gts[sources == 0], probs[sources == 0]
                ),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "val/strong_auroc",
                compute_binary_auroc(gts[sources != 0], probs[sources != 0]),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "val/strong_ap",
                compute_binary_average_precision(
                    gts[sources != 0], probs[sources != 0]
                ),
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )
        # epoch = self.current_epoch
        cls0 = probs[gts == 0]
        cls1 = probs[gts == 1]

        def safe_histogram(xs: np.ndarray, num_bins: int = 10):
            # 1) empty → return an empty histogram over [0,1]
            if xs.size == 0:
                # zero counts, two edges [0,1]
                return wandb.Histogram(
                    np_histogram=(np.array([0]), np.array([0.0, 1.0]))
                )

            # 2) all equal → force a tiny two-bin around that value
            if np.allclose(xs, xs.flat[0]):
                v = float(xs.flat[0])
                eps = 1e-6
                counts = np.array([len(xs), 0])
                edges = np.array([v - eps, v + eps, v + 2 * eps])
                return wandb.Histogram(np_histogram=(counts, edges))

            # 3) normal case
            return wandb.Histogram(xs.tolist(), num_bins=num_bins)

        # 4) log via your WandbLogger
        if isinstance(self.logger, WandbLogger):
            # this will overwrite the key in the run; you'll see a single table you can page through
            self.logger.experiment.log(
                {
                    "val/prob_dist/class0": safe_histogram(cls0),
                    "val/prob_dist/class1": safe_histogram(cls1),
                    "epoch": self.current_epoch,
                }
            )

        if self.train_step_losses:
            mean_class_loss = torch.stack(list(self.train_step_losses)).mean()
            self.log(
                "train/class_loss",
                mean_class_loss,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            tqdm.write(f"Train Classification loss: {mean_class_loss.item():.4f}")
            self.train_step_losses.clear()

        if self.train_step_supcon_losses:
            mean_supcon_loss = torch.stack(list(self.train_step_supcon_losses)).mean()
            self.log(
                "train/proj_loss",
                mean_supcon_loss,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            tqdm.write(f"Train Projection loss: {mean_supcon_loss.item():.4f}")
            self.train_step_supcon_losses.clear()

        if self.val_step_losses:
            self.log(
                "val/loss",
                torch.stack(list(self.val_step_losses)).mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            self.val_step_losses.clear()

    def _select_balanced_subset_indices(
        self, labels: np.ndarray, ids: Optional[List[str]]
    ):
        labels_int = np.asarray(labels).astype(int).reshape(-1)
        pos_idx = np.where(labels_int == 1)[0]
        neg_idx = np.where(labels_int == 0)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            return np.array([], dtype=int)

        n_pos = pos_idx.size
        # keep all positives and sample up to 2× negatives for better imbalance visibility
        desired_negs = int(2 * n_pos)
        n_neg = min(desired_negs, neg_idx.size)
        seed = (
            int(self.umap_seed)
            if self.umap_seed is not None
            else int(torch.initial_seed() % (2**32 - 1))
        )

        if ids is not None:
            neg_ids = np.asarray(ids, dtype=object)[neg_idx]
            neg_ids_sorted = np.sort(neg_ids.astype(str))
            rng = np.random.default_rng(seed)
            chosen_neg_ids = rng.permutation(neg_ids_sorted)[:n_neg]
            id_to_index: Dict[str, int] = {}
            for i, exam_id in enumerate(ids):
                if exam_id not in id_to_index:
                    id_to_index[exam_id] = i
            chosen_neg_idx = [
                id_to_index[i] for i in chosen_neg_ids if i in id_to_index
            ]
            chosen_neg_idx = np.asarray(chosen_neg_idx, dtype=int)
        else:
            rng = np.random.default_rng(seed)
            chosen_neg_idx = rng.permutation(neg_idx)[:n_neg]

        subset_idx = np.concatenate([pos_idx, chosen_neg_idx])
        return subset_idx

    def _log_umap_diagnostics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        ids: Optional[List[str]],
        sources: Optional[np.ndarray] = None,
        precomputed_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if hasattr(self, "trainer") and self.trainer is not None:
            if (
                hasattr(self.trainer, "is_global_zero")
                and not self.trainer.is_global_zero
            ):
                return
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected embeddings shaped [N,V,D] for UMAP, got {embeddings.shape}."
            )
        if embeddings.shape[1] < 1:
            raise ValueError(
                f"Expected at least one view in embeddings [N,V,D], got {embeddings.shape}."
            )
        if ids is None:
            raise ValueError(
                "UMAP logging requires 'exam_id' in the validation batch to build a stable subset."
            )
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Expected ids length {len(ids)} to match embeddings N={embeddings.shape[0]}."
            )

        # Pick a deterministic seed. If umap_seed is None, derive it from process seed.
        seed = (
            int(self.umap_seed)
            if self.umap_seed is not None
            else int(torch.initial_seed() % (2**32 - 1))
        )

        labels_int = np.asarray(labels).astype(int).reshape(-1)
        subset_indices = self._select_balanced_subset_indices(labels_int, ids)
        if subset_indices.size == 0:
            tqdm.write("Skipping UMAP logging (no balanced subset).")
            return

        # Always use view 0 as requested: [N,D]
        x = embeddings[np.asarray(subset_indices), 0, :].astype(np.float32, copy=False)
        y = labels_int[np.asarray(subset_indices)]

        # L2-normalize for stability (especially with cosine metric).
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.clip(norms, a_min=1e-12, a_max=None)

        try:
            import umap  # type: ignore
        except Exception as e:
            tqdm.write(f"UMAP not available ({repr(e)}); skipping UMAP logging.")
            return

        # Avoid interactive backends (e.g. TkAgg) in training loops / worker threads.
        # Build the figure without pyplot to prevent tkinter-related shutdown errors.
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
        from matplotlib.figure import Figure  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=int(self.umap_n_neighbors),
            min_dist=float(self.umap_min_dist),
            n_components=2,
            metric=str(self.umap_metric),
            n_epochs=int(self.umap_n_epochs),
            random_state=seed,
            n_jobs=1,
        )
        u2 = reducer.fit_transform(x)

        # Compute metrics on the exact subset being plotted if not provided.
        if precomputed_metrics is not None:
            subset_metrics = precomputed_metrics
        else:
            emb_subset = x.reshape(x.shape[0], 1, x.shape[1])
            try:
                subset_metrics = compute_representation_metrics(
                    emb_subset, y, max_samples=emb_subset.shape[0], seed=seed
                )
            except Exception as e:
                subset_metrics = {}
                tqdm.write(f"Failed computing subset metrics for UMAP: {repr(e)}")

        with (
            sns.axes_style("whitegrid"),
            sns.plotting_context("notebook", font_scale=0.9),
        ):
            # Increase width to accommodate the legend on the right without overflow.
            fig = Figure(figsize=(9.5, 8.5), dpi=200)
            canvas = FigureCanvas(fig)
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.35)
            ax_scatter = fig.add_subplot(gs[0, 0])

            # --- data prep ---
            label_map = {0: "healthy", 1: "chagas"}
            source_labels = {0: "CODE-15", 1: "PTB-XL", 2: "SaMi-Trop", -1: "unknown"}

            df = pd.DataFrame(
                {
                    "umap_x": u2[:, 0],
                    "umap_y": u2[:, 1],
                    "label": y.astype(int),
                }
            )
            df["label_name"] = df["label"].map(label_map)

            sources_arr_clean = None
            if sources is not None:
                arr = np.asarray(sources).reshape(-1)
                if arr.dtype.kind in {"f", "c"}:
                    arr = np.where(np.isnan(arr), -1, arr)
                sources_arr_clean = arr.astype(int)
                df["source"] = sources_arr_clean
                df["source_name"] = df["source"].map(
                    lambda v: source_labels.get(int(v), f"src {int(v)}")
                )
            else:
                df["source_name"] = "unknown"

            # Color = dataset, Marker = condition (label).
            # This makes 4 conditions (dataset×label) easier to distinguish without adding edgecolors.
            dataset_palette = {
                "CODE-15": "#377eb8",  # blue
                "PTB-XL": "#4daf4a",  # green
                "SaMi-Trop": "#ff7f00",  # orange
                "unknown": "#9aa0a6",  # gray
            }
            marker_map = {
                "healthy": "o",
                "chagas": "X",
            }

            # Shuffle the dataframe to prevent one group from masking others.
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

            background = "#f7f8fb"
            ax_scatter.set_facecolor(background)

            scatter_kwargs = {
                "data": df,
                "x": "umap_x",
                "y": "umap_y",
                "hue": "source_name",
                "palette": dataset_palette,
                "style": "label_name",
                "markers": marker_map,
                "s": 22,
                "edgecolor": None,
                "linewidth": 0,
                "alpha": 0.5,  # Balanced alpha for density
                "ax": ax_scatter,
                "legend": False,
            }
            sns.scatterplot(**scatter_kwargs)

            # Two compact legends: dataset colors and label markers.
            from matplotlib.lines import Line2D  # type: ignore

            dataset_order = [
                k
                for k in ("CODE-15", "PTB-XL", "SaMi-Trop")
                if k in set(df["source_name"])
            ]
            if "unknown" in set(df["source_name"]):
                dataset_order.append("unknown")
            handles_ds = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=dataset_palette[name],
                    markeredgecolor="none",
                    alpha=0.9,
                    label=name,
                )
                for name in dataset_order
            ]
            leg1 = ax_scatter.legend(
                handles=handles_ds,
                title="Dataset",
                loc="center left",
                bbox_to_anchor=(1.02, 0.62),
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#d4d4d4",
                fontsize=9,
                title_fontsize=10,
            )
            if leg1 is not None:
                leg1.get_title().set_fontweight("bold")
                ax_scatter.add_artist(leg1)

            handles_lbl = [
                Line2D(
                    [],
                    [],
                    marker=marker_map[name],
                    linestyle="",
                    markersize=7,
                    markerfacecolor="#111827" if marker_map[name] != "X" else "none",
                    markeredgecolor="#111827",
                    alpha=0.9,
                    label=name,
                )
                for name in ("healthy", "chagas")
            ]
            leg2 = ax_scatter.legend(
                handles=handles_lbl,
                title="Label",
                loc="center left",
                bbox_to_anchor=(1.02, 0.40),
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#d4d4d4",
                fontsize=9,
                title_fontsize=10,
            )
            if leg2 is not None:
                leg2.get_title().set_fontweight("bold")

            sns.despine(ax=ax_scatter, left=True, bottom=True)
            ax_scatter.set_xticks([])
            ax_scatter.set_yticks([])
            ax_scatter.set_xlabel("")
            ax_scatter.set_ylabel("")
            ax_scatter.set_title(
                f"UMAP view 0 — epoch {self.current_epoch} | track {self.track}",
                fontsize=12,
                weight="semibold",
                pad=10,
            )
            ax_scatter.text(
                0.01,
                0.98,
                f"samples={len(df)}   pos={int((y == 1).sum())}   neg={int((y == 0).sum())}",
                transform=ax_scatter.transAxes,
                va="top",
                fontsize=9.5,
                color="#1f2937",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#ffffff",
                    edgecolor="#d4d4d4",
                    linewidth=0.6,
                ),
            )

            # --- metrics panel ---
            ax_metrics = fig.add_subplot(gs[1, 0])
            ax_metrics.set_facecolor(background)
            ax_metrics.set_xticks([])
            ax_metrics.set_yticks([])
            sns.despine(ax=ax_metrics, left=True, bottom=True)

            lines = []
            if subset_metrics:
                metrics_order = ("SAD", "SAA", "CAD", "CAC", "GPU")
                # Define fixed widths for values to ensure vertical alignment across classes.
                # SAD and GPU are in [0,1], while SAA, CAD, and CAC can be larger.
                val_widths = {"SAD": 5, "SAA": 7, "CAD": 7, "CAC": 7, "GPU": 5}

                for cls in (0, 1):
                    parts = []
                    for m in metrics_order:
                        key = f"{m}_{cls}"
                        if key in subset_metrics:
                            val = subset_metrics[key]
                            w = val_widths.get(m, 7)
                            if (
                                np.isnan(val)
                                and m in ("SAD", "SAA")
                                and embeddings.shape[1] < 2
                            ):
                                parts.append(f"{m}={'N/A':>{w}}")
                            else:
                                parts.append(f"{m}={val:>{w}.3f}")
                    cls_label = "class 0" if cls == 0 else "class 1"
                    lines.append(f"{cls_label:<8} " + "   ".join(parts))

            metric_text = "\n".join(lines) if lines else "metrics unavailable"
            ax_metrics.text(
                0.02,
                0.5,
                metric_text,
                va="center",
                ha="left",
                fontsize=9,
                family="monospace",
                color="#111827",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#ffffff",
                    edgecolor="#d4d4d4",
                    linewidth=0.6,
                ),
            )
            # Adjust margins to ensure legend and metrics are fully visible.
            fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.78)

        # Render to an RGB array so wandb doesn't need to manage the figure backend.
        canvas.draw()
        rgba = np.array(canvas.buffer_rgba(), copy=True)  # type: ignore[attr-defined]
        image = np.asarray(rgba[..., :3], dtype=np.uint8)

        self.logger.experiment.log(  # type: ignore[union-attr]
            {
                "val/umap": wandb.Image(image),
                "epoch": self.current_epoch,
            },
            step=int(self.global_step),
        )

    def configure_optimizers(self):
        param_groups = split_optimizer_in_decay_and_no_decay(
            self,
            self.hparams.classifier_weight_decay,  # ty: ignore[unresolved-attribute]
            self.hparams.params_weight_decay,  # ty: ignore[unresolved-attribute]
        )
        optimizer_name = self.hparams.optimizer.lower()  # ty: ignore[unresolved-attribute]
        optimizer_kwargs: dict[str, Any] = {}
        if optimizer_name in ("adam", "adamw"):
            betas = getattr(self.hparams, "optimizer_betas", None)
            if betas is not None:
                optimizer_kwargs["betas"] = tuple(betas)
            eps = getattr(self.hparams, "optimizer_eps", None)
            if eps is not None:
                optimizer_kwargs["eps"] = eps

        optimizer = get_optimizer(
            name=self.hparams.optimizer,  # ty: ignore[unresolved-attribute]
            params=param_groups,
            lr=self.max_lr,
            weight_decay=self.hparams.params_weight_decay,  # ty: ignore[unresolved-attribute]
            momentum=getattr(self.hparams, "momentum", 0.0),
            **optimizer_kwargs,
        )

        # robustly fetch lr_scheduler from hparams whether hparams behaves like a mapping or an object
        if isinstance(self.hparams, dict):
            _scheduler_raw = self.hparams.get("lr_scheduler", None)
        else:
            _scheduler_raw = getattr(self.hparams, "lr_scheduler", None)
        scheduler_type = str(_scheduler_raw or "none").lower()

        if scheduler_type == "none":
            return optimizer

        if scheduler_type == "one_cycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                total_steps=(
                    int(self.trainer.estimated_stepping_batches)
                    if self.trainer.estimated_stepping_batches is not None
                    else None
                ),
                pct_start=getattr(self.hparams, "one_cycle_pct_start", 0.1),
                div_factor=getattr(self.hparams, "one_cycle_div_factor", 25.0),
                final_div_factor=getattr(self.hparams, "final_div_factor", 1e4),
            )
            interval = "step"
        elif scheduler_type == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.trainer.max_epochs)
                if self.trainer.max_epochs is not None
                else 1,
                eta_min=getattr(self.hparams, "lr", self.max_lr)
                / getattr(self.hparams, "final_div_factor", 1e4),
            )
            interval = "epoch"
        elif scheduler_type == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(self.hparams, "step_size", 10),
                gamma=getattr(self.hparams, "step_gamma", 0.1),
            )
            interval = "epoch"
        else:
            raise ValueError(f"Unknown scheduler: {_scheduler_raw}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": interval,
                "frequency": 1,
                "name": scheduler_type,
            },
        }


def resnet18(**kwargs) -> LitResNet18:
    return LitResNet18(**kwargs)
