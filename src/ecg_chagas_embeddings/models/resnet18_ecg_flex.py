from typing import Callable, List, Optional, Type, Union, Tuple


import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger


from ecg_chagas_embeddings.helper_code import compute_accuracy, compute_challenge_score
from ecg_chagas_embeddings.models.losses import SupConLoss, ConSupPrototypeLoss
from ecg_chagas_embeddings.utils import (
    get_optimizer,
    split_optimizer_in_decay_and_no_decay,
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
        self.downsample = downsample
        self.stride = stride
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
        self.downsample = downsample
        self.stride = stride
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


class LitResNet18NJ(pl.LightningModule):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
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
        classifier_weight_decay: float = 1e-5,
        params_weight_decay: float = 1e-5,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        channels=12,
        initial_kernel_size=7,
        initial_stride=2,
        initial_padding=3,
        stochastic_depth_prob=0.0,
        crop_size=2500,
        max_time_warp=0.15,
        criterion=nn.BCEWithLogitsLoss(),
        use_sup_con=False,
        use_prototypes=False,
        classifier_weight=1.0,
        sup_con_weight=0.05,
        sup_con_temp=0.07,
        dropout_rate=0.1,
        se_reduction=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer
        self._norm_layer = norm_layer
        self.norm_groups = norm_groups
        self.max_lr = lr
        self.lr_scheduler = lr_scheduler

        self.inplanes = inplanes
        self.dilation = 1
        self.channels = channels
        self.initial_kernel_size = initial_kernel_size
        self.initial_stride = initial_stride
        self.initial_padding = initial_padding
        self.stochastic_depth_prob = stochastic_depth_prob
        self.crop_size = crop_size
        self.max_time_warp = max_time_warp
        self.criterion = criterion
        self.se_reduction = se_reduction
        self.use_sup_con = use_sup_con
        self.use_prototypes = use_prototypes
        self.classifier_weight = classifier_weight
        self.sup_con_weight = sup_con_weight
        self.sup_con_temp = sup_con_temp
        self.fake_sup_dropout = nn.Dropout(p=0.1)
        self.fake_sup_noise_std = 0.01
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
        # keep string in wandb config and prevent overwriting
        self.save_hyperparameters(ignore=["criterion"])
        self.train_step_losses = []
        self.train_step_supcon_losses = []
        self.val_step_losses = []
        self.validation_step_outputs = []
        self._pred_rows = []

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
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

        self._stage_block_id = 0
        self._sd_prob = 0.0
        self._total_layers = sum(layers)

        self.layer1 = self._make_layer(
            block,
            norm_type,
            norm_groups,
            inplanes,
            layers[0],
            se_reduction=self.se_reduction,
        )
        self.layer2 = self._make_layer(
            block,
            norm_type,
            norm_groups,
            inplanes * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            se_reduction=self.se_reduction,
        )
        self.layer3 = self._make_layer(
            block,
            norm_type,
            norm_groups,
            inplanes * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            se_reduction=self.se_reduction,
        )
        self.layer4 = self._make_layer(
            block,
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
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        feat_dim = inplanes * 8 * block.expansion

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
        self.inplanes = planes * block.expansion
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
            D = self.projection_head[-1].out_features
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
        if self.use_sup_con and "ecg_views" in batch:
            x = batch["ecg_views"]  # [B,2,C,T]
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

        elif self.use_prototypes and "ecg_views" in batch:
            x = batch["ecg_views"]  # [B,2,C,T]
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
            x = batch["ecg"]  # [B, C, T]
            feats, proj, logits = self(x)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)  # [B]

            cls_loss = compute_cls_loss(logits)
            cls_loss = self.classifier_weight * cls_loss
            self.train_step_losses.append(cls_loss.detach())
            return cls_loss

    def validation_step(self, batch, batch_idx):
        signals = batch["ecg"]
        labels = batch["chagas"]
        ages = batch.get("age", None)
        sexes = batch.get("sex", None)
        sources = batch.get("source", None)
        ids = batch.get("exam_id", None)
        feats, proj, logits = self(signals)

        # print(f"Sources: {sources}")

        metadata = {
            "labels": labels,
            "source": sources,
            "exam_id": ids,
            "age": ages,
            "sex": sexes,
        }

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
        if type(self.criterion).__name__ == "SourceWeightedTopTverskyLoss":
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

        # if sources is not None:
        #    unique_sources, counts = np.unique(sources, return_counts=True)
        #    tqdm.write("Source counts:")
        #    for src, count in zip(unique_sources, counts):
        #        tqdm.write(f"  Source {src}: {count}")

        self.validation_step_outputs.clear()

        # num_gts_ones = np.sum(gts == 1.0)
        # tqdm.write(f"Positive cases in epoch: {num_gts_ones} of total {len(gts)}")

        acc = compute_accuracy(gts, preds)
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
            score = 0.0
            tqdm.write(
                f"Error in computing challenge score: {repr(e)}. Setting score to 0.0"
            )

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
            self.hparams.classifier_weight_decay,
            self.hparams.params_weight_decay,
        )
        optimizer = get_optimizer(
            name=self.hparams.optimizer,
            params=param_groups,
            lr=self.max_lr,
            weight_decay=self.hparams.params_weight_decay,
            momentum=getattr(self.hparams, "momentum", 0.0),
        )

        scheduler_type = self.hparams.lr_scheduler.lower()
        if scheduler_type == "none":
            return optimizer

        if scheduler_type == "one_cycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1e4,
            )
            interval = "step"
        elif scheduler_type == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.lr / self.hparams.final_div_factor,
            )
            interval = "epoch"
        elif scheduler_type == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.step_size,
                gamma=self.hparams.step_gamma,
            )
            interval = "epoch"
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": interval,
                "frequency": 1,
                "name": scheduler_type,
            },
        }


def resnet18(**kwargs) -> LitResNet18NJ:
    return LitResNet18NJ(BasicBlock, [2, 2, 2, 2], **kwargs)
