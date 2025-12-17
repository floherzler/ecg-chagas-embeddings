from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


from lightning.pytorch import LightningModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth
from tqdm import tqdm
import wandb
from lightning.pytorch.loggers import WandbLogger


from ecg_chagas_embeddings.helper_code import compute_accuracy, compute_challenge_score
from ecg_chagas_embeddings.models.losses import SupConLoss, ConSupPrototypeLoss
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
        use_prototypes=False,
        classifier_weight=1.0,
        sup_con_weight=0.05,
        sup_con_temp=0.07,
        dropout_rate=0.1,
        se_reduction=None,
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
        self.use_sup_con = use_sup_con  # ty: ignore[unresolved-attribute]
        self.use_prototypes = use_prototypes  # ty: ignore[unresolved-attribute]
        self.classifier_weight = classifier_weight  # ty: ignore[unresolved-attribute]
        self.sup_con_weight = sup_con_weight  # ty: ignore[unresolved-attribute]
        self.sup_con_temp = sup_con_temp  # ty: ignore[unresolved-attribute]
        self.fake_sup_dropout = nn.Dropout(p=0.1)
        self.fake_sup_noise_std = 0.01  # ty: ignore[unresolved-attribute]
        self.sup_con_loss = SupConLoss(
            temperature=self.sup_con_temp,
            contrast_mode="ALL_VIEWS",
            base_temperature=0.07,
            ratio_supervised_majority=0.0,
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
        logits = self.fc(feats)

        return feats, logits

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feats, logits = self._forward_impl(x)
        proj = self.projection_head(feats)
        return feats, proj, logits

    def on_fit_start(self):
        if not self.use_prototypes:
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

            # classifier wants FLOAT targets
            cls_loss = self.criterion(logits, y_float)
            cls_loss = self.classifier_weight * cls_loss

            # SupCon can use INT labels
            with torch.cuda.amp.autocast(enabled=False):
                con_loss, *_ = self.sup_con_loss(proj.float(), y_long)
            con_loss = self.sup_con_weight * con_loss

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

            # classifier wants FLOAT targets
            cls_loss = self.criterion(logits, y_float)
            cls_loss = self.classifier_weight * cls_loss

            # prototype loss wants ONE-HOT (FLOAT)
            y_oh = F.one_hot(y_long, num_classes=2).to(torch.float32)
            with torch.cuda.amp.autocast(enabled=False):
                proto_loss, *_ = self.proto_loss(proj.float(), y_oh)
            proto_loss = self.sup_con_weight * proto_loss  # or self.proto_weight

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
            cls_loss = self.classifier_weight * cls_loss
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
        if type(self.criterion).__name__ == "SourceWeightedBCE":
            loss = self.criterion(logits, labels, metadata)
        elif type(self.criterion).__name__ == "SourceWeightedTopTverskyLoss":
            loss = self.criterion(logits, labels, metadata)
        else:
            loss = self.criterion(logits, labels)
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
        # ids = (
        #     sum((b[6] for b in self.validation_step_outputs), [])
        #     if self.validation_step_outputs[0][6] is not None
        #     else None
        # )
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

        try:
            emb_metrics = compute_representation_metrics(embeddings, gts)
            for k, v in emb_metrics.items():
                tqdm.write(f"Embedding metric {k}: {v:.4f}")
                self.log(f"emb_{k}", v, prog_bar=False, on_epoch=True, on_step=False)
        except Exception as e:
            tqdm.write(f"Error in computing representation metrics: {repr(e)}")

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

        if not len(self.train_step_losses) == 0 and not len(self.val_step_losses) == 0:
            self.log(
                "train/class_loss",
                torch.stack([x for x in self.train_step_losses]).mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            tqdm.write(
                f"Train Classification loss: {torch.stack([x for x in self.train_step_losses]).mean().item():.4f}"
            )
            if self.use_sup_con or self.use_prototypes:
                tqdm.write(
                    f"Train SupCon loss: {torch.stack([x for x in self.train_step_supcon_losses]).mean().item():.4f}"
                )
                self.log(
                    "train/sup_con_loss",
                    torch.stack([x for x in self.train_step_supcon_losses]).mean(),
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
                self.train_step_supcon_losses.clear()
        self.log(
            "val/loss",
            torch.stack([x for x in self.val_step_losses]).mean(),
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )
        self.train_step_losses.clear()
        self.val_step_losses.clear()

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
