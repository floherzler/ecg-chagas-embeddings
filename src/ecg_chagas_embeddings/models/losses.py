import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import numpy as np
from pytorch_lightning import LightningModule
import torch.distributed as dist
import torch.distributed.nn.functional as dist_f


class RankingBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        pos = inputs[targets == 1]
        neg = inputs[targets == 0]

        if len(pos) == 0 or len(neg) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Compute the difference between positive and negative samples
        diff = pos.view(-1, 1) - neg.view(1, -1)
        return F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))


class SourceWeightedBCE(nn.Module):
    def __init__(self, source_weights: dict, reduction="mean"):
        """
        Args:
            source_weights (dict or list): mapping from source index (int/float) to weight (float).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(SourceWeightedBCE, self).__init__()
        self.source_weights = source_weights
        self.reduction = reduction

    def forward(self, inputs, targets, metadata: dict):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        sources = metadata["source"]  # tensor shape (B, 1) or (B,)
        if isinstance(sources, list):
            sources = torch.tensor(sources, dtype=torch.long, device=inputs.device)
        elif isinstance(sources, torch.Tensor):
            sources = sources.long().view(-1)
        else:
            raise TypeError("metadata['source'] must be list or tensor")

        # Build weights from numeric source indices
        # If source_weights is a dict, you can access it with brackets using the key.
        # Here, we convert the source to string to match the dict keys.
        # tqdm.write(f"Sources: {sources}")
        weights = torch.tensor(
            [self.source_weights.get(int(src.item()), 1.0) for src in sources],
            dtype=bce_loss.dtype,
            device=inputs.device,
        )

        weighted_loss = bce_loss * weights

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.initial_gamma = gamma
        self.gamma = gamma
        self.reduction = reduction

    def update_gamma(self, new_gamma):
        self.gamma = new_gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss For Multi-Label Classification https://github.com/Alibaba-MIIL/ASL
    The loss dynamically down-weights and hard-thresholds easy negative samples,
    while also discarding possibly mislabeled samples.
    """

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        """
        Args:
            gamma_neg: Negative focusing parameter. Larger values have increased ignorance towards easy examples. Defaults to 4.
            gamma_pos: Positive focusing parameter. Larger values have increased ignorance towards easy examples. Defaults to 1.
            clip: Asymmetric Clipping of negative examples. Defaults to 0.05.
            eps: Prevents log of 0. Defaults to 1e-8.
            disable_torch_grad_focal_loss: Should gradient flow through focal loss. Defaults to False.
        """
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - Tversky


class TverskyTopLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        top_percent: float = 0.05,
        tau: float = 0.1,
        use_bce: bool = False,
        bce_weight: float = 0.0,
        use_focal: bool = False,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
    ):
        """
        top_percent: fraction of examples to treat as "selected" (e.g. 0.05 for 5%)
        tau: temperature for soft thresholding
        use_bce / bce_weight: add standard BCE(probs, targets)
        use_focal / focal_weight / focal_gamma: add focal loss on probs
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.top_percent = top_percent
        self.tau = tau

        self.use_bce = use_bce
        self.bce_weight = bce_weight

        self.use_focal = use_focal
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, metadata: dict
    ) -> torch.Tensor:
        # 1) get probabilities
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        # 2) compute dynamic threshold q for the top_percent
        N = probs.numel()
        k = max(1, int(self.top_percent * N))
        topk_vals, _ = probs.topk(k)
        q = topk_vals.min()  # the k-th largest probability

        # 3) soft‐mask of "selected" examples
        m = torch.sigmoid((probs - q) / self.tau)

        # 4) Tversky components
        TP = (m * targets).sum()
        FP = (m * (1 - targets)).sum()
        FN = ((1 - m) * targets).sum()

        tversky_coeff = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        loss = 1.0 - tversky_coeff

        # 5) optional BCE
        if self.use_bce and self.bce_weight > 0:
            bce = F.binary_cross_entropy(probs, targets)
            loss = loss + self.bce_weight * bce

        # 6) optional Focal
        if self.use_focal and self.focal_weight > 0:
            # pt = p if y=1, else 1-p
            pt = torch.where(targets == 1, probs, 1 - probs)
            # focal term applied to standard BCE
            bce_elem = F.binary_cross_entropy(probs, targets, reduction="none")
            focal_term = ((1 - pt) ** self.focal_gamma) * bce_elem
            loss = loss + self.focal_weight * focal_term.mean()

        return loss


class SourceWeightedTopTverskyLoss(nn.Module):
    def __init__(
        self,
        source_weights: dict,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        top_percent: float = 0.05,
        tau: float = 0.1,
        use_bce: bool = False,
        bce_weight: float = 0.0,
        use_focal: bool = False,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.source_weights = source_weights

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.top_percent = top_percent
        self.tau = tau

        self.use_bce = use_bce
        self.bce_weight = bce_weight

        self.use_focal = use_focal
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma

        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, metadata: dict):
        # --- build per-sample weights
        sources = metadata["source"]
        if isinstance(sources, list):
            sources = torch.tensor(sources, device=logits.device)
        else:
            sources = sources.long().view(-1).to(logits.device)

        # --- flatten logits and targets to shape [B]
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()

        # --- get per-sample weights w[i]
        w = torch.tensor(
            [self.source_weights.get(int(src.item()), 1.0) for src in sources],
            dtype=logits_flat.dtype,
            device=logits_flat.device,
        )

        # --- compute probabilities for ranking/focal
        probs_flat = torch.sigmoid(logits_flat)

        # --- determine the 5% soft‐threshold mask
        N = probs_flat.numel()
        k = max(1, int(self.top_percent * N))
        topk_vals, _ = probs_flat.topk(k)
        q = topk_vals.min()
        m = torch.sigmoid((probs_flat - q) / self.tau)

        # --- weighted Tversky terms
        TP = (w * targets_flat * m).sum()
        FP = (w * (1 - targets_flat) * m).sum()
        FN = (w * targets_flat * (1 - m)).sum()

        tversky_coeff = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        loss = 1.0 - tversky_coeff  # scalar

        # --- optional weighted BCE-with-logits
        if self.use_bce and self.bce_weight > 0:
            bce_elem = F.binary_cross_entropy_with_logits(
                logits_flat, targets_flat, reduction="none"
            )
            bce = (w * bce_elem).sum() / (w.sum() + 1e-12)
            loss = loss + self.bce_weight * bce

        # --- optional weighted focal
        if self.use_focal and self.focal_weight > 0:
            bce_elem = F.binary_cross_entropy_with_logits(
                logits_flat, targets_flat, reduction="none"
            )
            pt = torch.where(targets_flat == 1, probs_flat, 1 - probs_flat)
            focal_elem = ((1 - pt) ** self.focal_gamma) * bce_elem
            focal = (w * focal_elem).sum() / (w.sum() + 1e-12)
            loss = loss + self.focal_weight * focal

        # --- final reduction: here loss is already a scalar, so mean/sum are same
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss with optional Top-K selection, soft mask, entropy regularizer,
    and an auxiliary BCE-with-logits anchor.

    Args:
        alpha (float): weight of false positives
        beta (float): weight of false negatives
        gamma (float): focal exponent
        smooth (float): smoothing constant
        k (int|float|None): if float∈(0,1), fraction for top‐k; if int>1, absolute top‐k; if None, no selection
        tau (float): temperature for soft‐mask (higher → softer)
        entropy_weight (float): weight for the entropy regularizer
        bce_weight (float): weight for the auxiliary BCE-with-logits loss
        pos_weight (float|None): positive-class weight for BCE-with-logits (e.g. (1-p)/p for p=0.1 → 9.0)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 0.1,
        k: Union[int, float, None] = None,
        tau: float = 0.5,
        entropy_weight: float = 0.005,
        bce_weight: float = 0.0,
        pos_weight: Union[float, None] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.k = k
        self.tau = tau
        self.entropy_weight = entropy_weight
        self.bce_weight = bce_weight
        if pos_weight is not None:
            self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # raw logits → probabilities
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1).float()

        # build soft‐mask for top-k selection if requested
        if self.k is not None:
            N = probs.numel()
            if isinstance(self.k, float) and 0 < self.k < 1:
                top_k = max(1, int(self.k * N))
            else:
                top_k = int(self.k)
            topk_vals, _ = probs.topk(top_k)
            threshold = topk_vals.min()
            m = torch.sigmoid((probs - threshold) / self.tau)
        else:
            m = torch.ones_like(probs)

        # True Positives, False Positives & False Negatives (soft‐masked)
        TP = (m * probs * targets).sum()
        FP = (m * probs * (1 - targets)).sum()
        FN = (m * (1 - probs) * targets).sum()

        # Tversky coefficient + focal exponent
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        ft_loss = (1 - tversky) ** self.gamma

        # entropy regularizer (penalize uncertainty)
        eps = 1e-6
        entropy = -(
            probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)
        ).mean()
        # encourage more entropy
        loss = ft_loss - self.entropy_weight * entropy

        # auxiliary BCE-with-logits to keep background gradients alive
        if self.bce_weight > 0:
            bce = self.bce_fn(inputs.view(-1), targets)
            loss = loss + self.bce_weight * bce

        return loss


"""
Loss functions for contrastive learning on imbalanced binary datasets.

This module contains implementations of various contrastive loss functions used in the paper.
"""


class SupConLoss(LightningModule):
    """
    Supervised Contrastive Learning loss.

    Extends contrastive learning to use label information, pulling together
    samples from the same class while pushing apart samples from different classes.

    This implementation includes special handling for imbalanced binary classification
    with mechanisms to address class imbalance.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "ALL_VIEWS",
        base_temperature: float = 0.07,
        min_class: Optional[int] = None,
        ratio_supervised_majority: float = -1.0,
        weighting_positives: float = -1.0,
        reweight_global_min_loss: float = -1.0,
    ) -> None:
        """
        Initialize the Supervised Contrastive loss function.

        Args:
            temperature: Temperature scaling parameter for the logits
            contrast_mode: Mode for contrasting ("ALL_VIEWS" uses all views as anchors)
            base_temperature: Base temperature for scaling
            min_class: Minority class index (0 or 1)
            ratio_supervised_majority (float, optional): Fraction of majority class positive pairs to *include*
            for supervision. Defaults to -1.0.
            - Values range from 0.0 (no majority-majority positive pairs) to 1.0 (all
              majority-majority pairs are considered positive).
            - A value of -1.0 (default) results in standard supervised contrastive
              loss, where all same-class pairs (including all majority-majority)
              are considered positive.
            - This parameter is active only if labels are provided and its value is >= 0.0.
            weighting_positives: Weight for positive samples
            reweight_global_min_loss: Weight for global minority loss reweighting
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        if min_class is None:
            raise ValueError("min_class must be specified (0 or 1)")

        self.majority_class = 0 if int(min_class) == 1 else 1
        self.minority_class = int(min_class)
        self.ratio_supervised_majority = ratio_supervised_majority
        self.weighting_positives = weighting_positives
        self.reweight_global_min_loss = reweight_global_min_loss

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute the Supervised Contrastive loss.

        Args:
            features: Tensor of shape [bsz, n_views, ...] containing the embeddings
                     of the samples and their augmentations.
            labels: Tensor of shape [bsz] containing class labels (0 or 1).

        Returns:
            If only the loss is computed: The computed loss value
            Otherwise: Tuple of (loss, logits, batch_labels)
        """
        # Validate input dimensions
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Convert labels to tensor if needed
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        # Initialize masks
        mask_unsup = None
        mask_maj = None
        mask_min = None

        # Create identity matrix for unsupervised case
        mask_unsup = torch.eye(batch_size, dtype=torch.float32).to(self.device)

        # Handle different modes based on label availability and configuration
        if labels is None:
            # SimCLR unsupervised loss
            mask = mask_unsup
        elif 0.0 <= self.ratio_supervised_majority < 1.0:
            labels = labels.contiguous().view(-1, 1) if labels is not None else None

            # Ensure labels dimensions match expected shapes
            if labels is not None and labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features")

            # SupCon with fewer positives in majority class
            numerator_mask, ratio = self._create_numerator_mask(
                labels,
                self.ratio_supervised_majority,
                self.minority_class,
                self.majority_class,
            )
            mask = numerator_mask
        else:
            # Standard supervised contrastive mode
            min_indices = labels == self.minority_class
            maj_indices = labels == self.majority_class

            labels = labels.contiguous().view(-1, 1) if labels is not None else None

            # Ensure labels dimensions match expected shapes
            if labels is not None and labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features")

            # Create mask based on class labels (same class = positive pair)
            mask = torch.eq(labels, labels.T).float().to(self.device)

            # Create separate masks for majority and minority classes
            mask_maj, mask_min = mask.clone(), mask.clone()
            mask_maj[min_indices] = 0.0
            mask_maj[:, min_indices] = 0.0
            mask_min[maj_indices] = 0.0
            mask_min[:, maj_indices] = 0.0

        # Combine features from all views
        contrast_count = features.shape[1]  # Number of views/augmentations
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "ALL_VIEWS":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Calculate similarity matrix (logits)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # Subtract max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat masks for all views
        mask = mask.repeat(anchor_count, contrast_count)
        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)
        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
        if mask_unsup is not None:
            mask_unsup = mask_unsup.repeat(anchor_count, contrast_count)

        # Create mask to exclude self-contrasts
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,  # value to fill
        )

        # Apply the self-contrast mask
        mask = mask * logits_mask
        if mask_unsup is not None:
            mask_unsup = mask_unsup * logits_mask

        if mask_min is not None:
            mask_maj = mask_maj * logits_mask
            mask_min = mask_min * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-8)

        # Scale by temperature if base_temperature is set
        if self.base_temperature > 0.0:
            loss = -self.temperature / self.base_temperature * mean_log_prob_pos
        else:
            loss = -mean_log_prob_pos

        # Average across all anchors
        loss = loss.view(anchor_count, batch_size).mean()

        # Prepare labels for computing metrics
        labels = torch.arange(batch_size, device=self.device, dtype=torch.long)
        labels = labels + batch_size - 1  # Remove sim to self
        labels = torch.cat(
            [labels, torch.arange(batch_size, device=self.device, dtype=torch.long)],
            dim=0,
        )

        # Remove self-similarity from logits
        clean_logits = exp_logits[~torch.eye(batch_size * anchor_count).bool()].view(
            batch_size * anchor_count, -1
        )

        return loss, clean_logits, labels

    def _create_numerator_mask(
        self,
        labels: torch.Tensor,
        ratio: float,
        minority_class: int,
        majority_class: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Create a mask for the numerator that includes all minority class samples
        and a controlled ratio of majority class samples.

        Args:
            labels: Class labels for each sample.
            ratio: Fraction of majority class samples whose positive pairs (with other majority
                   class samples) should be included in the mask. Must be between 0.0 and 1.0.
                   0.0 means no majority-majority positive pairs.
                   1.0 means all potential majority-majority positive pairs are included.
            minority_class: Index of the minority class.
            majority_class: Index of the majority class.

        Returns:
            Mask for numerator and effective ratio
        """

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Ratio must be between 0 and 1, but is {ratio}")

        # Get indices for each class
        minority_indices = torch.where(labels == minority_class)[0]
        majority_indices = torch.where(labels == majority_class)[0]

        # Select subset of majority class samples
        selected_majority_indices = torch.randperm(majority_indices.size(0))[
            : int(ratio * majority_indices.size(0))
        ]
        selected_majority_indices = majority_indices[selected_majority_indices]

        # Combine minority and selected majority indices
        combined_indices = torch.cat([minority_indices, selected_majority_indices])

        # Create mask for selected samples
        selected_labels = labels[combined_indices].view(-1, 1)
        mask_partial = (selected_labels == selected_labels.T).float()

        # Final mask of size labels.size(0) x labels.size(0)
        mask = torch.zeros(labels.size(0), labels.size(0), dtype=torch.float32).to(
            self.device
        )
        mask[combined_indices.reshape(-1, 1), combined_indices] = mask_partial

        # Add diagonal (self-pairs)
        mask += torch.eye(mask.size(0)).to(self.device)
        mask.clamp_(max=1)

        return mask, ratio


class SupConLossKCLTSC(nn.Module):
    """Targeted Supervised Contrastive Learning for Long-Tailed Recognition (TSC): https://arxiv.org/pdf/2004.11362.pdf.
    EXPLORING BALANCED FEATURE SPACES FOR REPRESENTATION LEARNING (KCL): https://openreview.net/pdf?id=OqtLIabPTit"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        use_tcl=False,
        k=3,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        prototypes = self.generate_random_vector_and_negative(
            128
        )  # Expected shape: [num_classes, feature_dim]
        self.register_buffer("prototypes", prototypes)
        self.k = k
        self.use_tcl = use_tcl

        print(
            f"SupConLossTCL {self.temperature} {self.contrast_mode} {self.base_temperature} {self.use_tcl} {self.k}"
        )

    def generate_random_vector_and_negative(self, d):
        """
        Generates a random vector on the d-dimensional unit hypersphere and its negative.

        Parameters:
        - d (int): The dimension of the hypersphere.

        Returns:
        - torch.Tensor: A 2 x d array where the first row is the random vector
                        and the second row is its negative.
        """
        random_vector = np.random.randn(d)
        norm = np.linalg.norm(random_vector)
        if norm == 0:
            raise ValueError("Generated a zero vector, which cannot be normalized.")
        unit_vector = random_vector / norm
        negative_vector = -unit_vector
        vectors = np.vstack((unit_vector, negative_vector))
        return torch.from_numpy(vectors).float()

    def forward(
        self,
        features,
        labels=None,
        mask=None,
    ):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        k = self.k
        tcl = self.use_tcl
        num_prototypes = 2

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        anchor_labels = torch.cat([labels, labels], dim=0)
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        mask_augs = torch.eye(batch_size, dtype=torch.float32).to(device)
        # print(anchor_labels.shape)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = torch.cat([contrast_feature, self.prototypes], dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits = logits[:-2, :]

        mask = mask.repeat(anchor_count, contrast_count)
        mask_augs = mask_augs.repeat(anchor_count, contrast_count)

        prototype_mask = torch.zeros((2 * batch_size, num_prototypes), device=device)
        prototype_mask[torch.arange(2 * batch_size), anchor_labels] = 1

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        mask_augs = mask_augs * logits_mask

        mask_noself = mask * logits_mask

        augmentations_mask = mask_augs.float()

        mask_noself = mask_noself - augmentations_mask.float()

        mask[mask < 1e-6] = 0

        # Randomly select up to k positives per anchor
        num_pos = mask_noself.sum(dim=1)
        # k_max = min(k, num_pos.max().int().item())

        # For anchors with fewer than k positives, adjust k accordingly
        new_pos_mask = torch.zeros_like(mask)
        for i in range(2 * batch_size):
            k_i = min(k, int(num_pos[i].item()))

            pos_indices = torch.nonzero(mask_noself[i]).squeeze()
            if len(pos_indices) > 0:
                selected_indices = pos_indices[torch.randperm(len(pos_indices))[:k_i]]
                new_pos_mask[i, selected_indices] = 1

        new_pos_mask = augmentations_mask.float() + new_pos_mask.float()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(new_pos_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        mask = new_pos_mask * logits_mask

        if tcl:
            mask = torch.cat([mask, prototype_mask.float()], dim=1)

        mask = mask.clamp(max=1)
        if tcl:
            logits_mask = torch.cat(
                [logits_mask, torch.ones(2 * batch_size, 2).float().to(device)], dim=1
            )
        if not tcl:
            logits = logits[:, :-2]
        # logits =logits[:,:-2]

        print(prototype_mask.sum(1).mean().cpu().numpy())

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        print(mask_pos_pairs.mean().cpu().numpy())
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ConSupPrototypeLoss(LightningModule):
    """
    Supervised Contrastive Learning with Prototypes.
    This loss function extends supervised contrastive
    learning by introducing class prototypes,
    which are representative vectors for each class.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "ALL_VIEWS",
        base_temperature: float = 0.07,
        negatives_weight: float = 1.0,
        eps: float = 0.1,
        eps_0: Optional[float] = None,
        eps_1: Optional[float] = None,
        minority_cls: Optional[int] = None,
        max_epoch: Optional[int] = None,
    ) -> None:
        """
        Initialize the Supervised Contrastive with Prototypes loss function.

        Args:
            temperature: Temperature scaling parameter for the logits
            contrast_mode: Mode for contrasting ("ALL_VIEWS" or "ONE_VIEW")
            base_temperature: Base temperature for scaling
            negatives_weight: Weight for negative samples
            eps: Margin parameter for prototype distance
            eps_0: Class-specific margin for class 0
            eps_1: Class-specific margin for class 1
            minority_cls: Index of the minority class
            max_epoch: Maximum number of training epochs
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.eps = eps
        self.eps_0 = eps_0
        self.eps_1 = eps_1

        self.prototypes = None
        self.negatives_weight = negatives_weight

        # Use same epsilon for both classes if they're equal
        if eps_0 is not None and eps_0 == eps_1:
            self.eps = self.eps_0 = self.eps_1 = eps_0

        self.minority_cls = minority_cls
        self.max_epoch = max_epoch

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the class prototypes to use in the loss calculation."""
        self.prototypes = prototypes.to(self.device)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Supervised Contrastive with Prototypes loss.

        Args:
            features: Tensor of shape [bsz, n_views, ...] containing the embeddings
                     of the samples and their augmentations.
            labels: Tensor of shape [bsz, 2] where each row is a one-hot encoding
                   of the class label.

        Returns:
            Tuple of (loss, logits, batch_labels)
        """
        # Validate input dimensions
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Verify prototypes are available and correctly configured
        num_prototypes = 0
        if self.prototypes is not None:
            num_prototypes = self.prototypes.size(0)
            if num_prototypes != 2:
                raise ValueError("Number of prototypes must be 2")

        if self.prototypes.device != self.device:
            self.prototypes = self.prototypes.to(self.device)

        # Initialize masks
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask_maj = mask_min = mask_sup = None

        # Process labels if provided
        if labels is not None:
            # Extract class dimension and validate shapes
            labels_dim = labels[:, 1].contiguous().view(-1, 1)
            if labels_dim.shape != (batch_size, 1) or labels.shape != (batch_size, 2):
                raise ValueError("Number of labels does not match number of features")

            # Create mask based on label equality (positive pairs)
            mask_sup = torch.eq(labels_dim, labels_dim.T).float().to(self.device)
            labels_dim = labels_dim.squeeze(1)

            # Create boolean indices for each class
            indices_0 = (labels_dim == 0).to(self.device)
            indices_1 = (labels_dim == 1).to(self.device)

            # Initialize class-specific masks
            mask_0, mask_1 = mask_sup.clone(), mask_sup.clone()

            # Set opposite class indices to 0 in each mask
            mask_0[indices_1] = 0
            mask_0[:, indices_1] = 0
            mask_1[indices_0] = 0
            mask_1[:, indices_0] = 0

            # Validate minority class is defined
            if self.minority_cls is None:
                raise ValueError("minority_cls must be set")

            # Assign masks based on minority class
            mask_min, mask_maj = (
                (mask_0, mask_1) if self.minority_cls == 0 else (mask_1, mask_0)
            )

        # Number of views/augmentations
        contrast_count = features.shape[1]

        # Combine features from all views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Add prototypes to features if available
        if self.prototypes is not None:
            contrast_feature = torch.cat([contrast_feature, self.prototypes], dim=0)

        # Set up anchor features based on contrast mode
        if self.contrast_mode == "ONE_VIEW":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "ALL_VIEWS":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Calculate similarity matrix
        sims = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(sims, self.temperature)

        # Subtract max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat mask for all views
        mask = mask.repeat(anchor_count, contrast_count)

        # Create mask to exclude self-contrasts
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,  # value to fill
        )

        # Apply masks for different classes if available
        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)
            mask_maj = mask_maj * logits_mask

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
            mask_min = mask_min * logits_mask

        if mask_sup is not None:
            mask_sup = mask_sup.repeat(anchor_count, contrast_count)
            mask_sup = mask_sup * logits_mask

        # Apply self-contrast mask
        mask = mask * logits_mask

        # Handle prototype-based loss
        if self.prototypes is not None:
            if labels is None:
                raise ValueError("Labels must be provided if prototypes are provided")

            bsz = labels.shape[0]
            selected_prototypes = labels
            selected_prototypes_mask = selected_prototypes.to(self.device)

            assert selected_prototypes_mask.shape[0] == bsz
            assert selected_prototypes_mask.shape[1] == num_prototypes

            # Repeat for second view
            selected_prototypes_mask = torch.cat(
                [selected_prototypes_mask, selected_prototypes_mask], dim=0
            )

            # Extend logits mask for prototypes
            logits_mask = torch.cat(
                [
                    logits_mask,
                    torch.zeros_like(selected_prototypes_mask).to(self.device),
                ],
                dim=1,
            )

            # Remove prototype anchors
            logits = logits[:-num_prototypes, :]

        # Calculate exponential of logits and apply mask
        exp_logits = torch.exp(logits) * logits_mask

        # Compute log probability
        log_prob = logits - torch.log(
            self.negatives_weight * exp_logits.sum(1, keepdim=True)
        )

        # Apply prototype pull mechanism if prototypes are available
        if self.prototypes is not None:
            # Initialize pull mask
            m_pull = torch.ones_like(selected_prototypes_mask, dtype=torch.float32).to(
                self.device
            )

            # Determine which samples to pull based on similarity and margin
            cond2 = sims[:-num_prototypes, -1] <= (
                sims[:-num_prototypes, -2] + self.eps_1
            )
            cond1 = sims[:-num_prototypes, -2] <= (
                sims[:-num_prototypes, -1] + self.eps_0
            )

            # Apply conditions and create final prototype mask
            p2_mask = torch.logical_and(cond2, selected_prototypes_mask[:, -1].bool())
            p1_mask = torch.logical_and(cond1, selected_prototypes_mask[:, -2].bool())
            selected_prototypes_mask = torch.stack((p1_mask, p2_mask), dim=1).float()

            # Apply the mask
            m_pull = m_pull * selected_prototypes_mask
            mask = torch.cat([mask, m_pull], dim=1)

        # Calculate number of positive pairs for each anchor
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        # Calculate mean log probability over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Apply temperature scaling
        if self.base_temperature > 0.0:
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        else:
            loss = -1.0 * mean_log_prob_pos

        # Average loss across all samples
        loss = loss.mean()

        # Prepare labels for metrics
        labels = torch.arange(batch_size, device=self.device, dtype=torch.long)
        labels = labels + batch_size - 1  # Remove sim to self
        labels = torch.cat(
            [labels, torch.arange(batch_size, device=self.device, dtype=torch.long)],
            dim=0,
        )

        # Remove prototypes from logits if used
        if self.prototypes is not None:
            exp_logits = exp_logits[:, :-num_prototypes]

        # Remove self-similarity from logits
        clean_logits = exp_logits[~torch.eye(batch_size * anchor_count).bool()].view(
            batch_size * anchor_count, -1
        )

        return loss, clean_logits.float().detach(), labels.detach()


class SupConLossDDP(nn.Module):
    """
    Supervised contrastive loss function for distributed data parallel (DDP) training.

    This implementation is optimized for distributed training and includes special handling
    for imbalanced binary classification with mechanisms to address class imbalance.

     Args:
        temperature (float): Temperature scaling parameter for the logits.
        contrast_mode (str): Mode for contrasting ("ALL_VIEWS" uses all views as anchors).
        base_temperature (float): Base temperature for scaling.
        min_class (Optional[int]): Minority class index (0 or 1).
        ratio_supervised_majority (float, optional): Fraction of majority class positive pairs to include
            for supervision. Defaults to -1.0.
            - Values range from 0.0 (no majority-majority positive pairs) to 1.0 (all
              majority-majority pairs are considered positive).
            - A value of -1.0 (default) results in standard supervised contrastive
              loss, where all same-class pairs (including all majority-majority)
              are considered positive.
            - This parameter is active only if labels are provided and its value is >= 0.0.
        weighting_positives (float): Weight for positive samples. (Currently not implemented in loss calculation)
        reweight_global_min_loss (float): Weight for global minority loss reweighting. (Currently not implemented in loss calculation)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "ALL_VIEWS",
        base_temperature: float = 0.07,
        min_class: Optional[int] = None,
        ratio_supervised_majority: float = -1.0,
        weighting_positives: float = -1.0,
        reweight_global_min_loss: float = -1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        if min_class is None:
            raise ValueError("min_class must be specified (0 or 1)")

        self.majority_class = 0 if int(min_class) == 1 else 1
        self.minority_class = int(min_class)
        self.ratio_supervised_majority = ratio_supervised_majority
        self.weighting_positives = weighting_positives
        self.reweight_global_min_loss = reweight_global_min_loss

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Supervised Contrastive loss with DDP support.

        Args:
            features: Tensor of shape [bsz, n_views, ...] containing the embeddings
                     of the samples and their augmentations.
            labels: Tensor of shape [bsz] containing class labels (0 or 1).

        Returns:
            Tuple of (loss, logits, batch_labels)
        """
        # Validate input dimensions
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device

        # Gather features and labels from all devices if using DDP
        if dist.is_available() and dist.is_initialized():
            gathered_features = dist_f.all_gather(features)
            features = torch.cat(gathered_features, dim=0)

            if labels is not None:
                gathered_labels = dist_f.all_gather(labels)
                labels = torch.cat(gathered_labels, dim=0)

        batch_size = features.shape[0]

        # Initialize masks
        mask_unsup = None
        mask_maj = None
        mask_min = None

        # Create identity matrix for unsupervised case
        mask_unsup = torch.eye(batch_size, dtype=torch.float32).to(device)

        # Handle different modes based on label availability and configuration
        if labels is None:
            # SimCLR unsupervised loss
            mask = mask_unsup
        elif 0.0 <= self.ratio_supervised_majority < 1.0:
            # Format labels as a column vector
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float32).to(device)

            labels = labels.contiguous().view(-1, 1)

            # Ensure labels dimensions match expected shapes
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features")

            # SupCon with fewer positives in majority class
            numerator_mask, ratio = self._create_numerator_mask(
                labels,
                self.ratio_supervised_majority,
                self.minority_class,
                self.majority_class,
                device=device,
            )
            mask = numerator_mask
        else:
            # Format labels as needed
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float32).to(device)

            # Create masks for supervised loss with majority/minority separation
            min_indices = labels == self.minority_class
            maj_indices = labels == self.majority_class

            labels = labels.contiguous().view(-1, 1)

            # Ensure labels dimensions match expected shapes
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features")

            # Create mask based on class labels (same class = positive pair)
            mask = torch.eq(labels, labels.T).float().to(device)
            mask_maj, mask_min = mask.clone(), mask.clone()

            # Create separate masks for majority and minority classes
            mask_maj[min_indices] = 0.0
            mask_maj[:, min_indices] = 0.0
            mask_min[maj_indices] = 0.0
            mask_min[:, maj_indices] = 0.0

        # Number of views/augmentations
        contrast_count = features.shape[1]

        # Combine features from all views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "ALL_VIEWS":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # Calculate similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # Subtract max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat masks for all views
        mask = mask.repeat(anchor_count, contrast_count)
        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)

        if mask_unsup is not None:
            mask_unsup = mask_unsup.repeat(anchor_count, contrast_count)

        # Create mask to exclude self-contrasts
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,  # value to fill
        )

        # Apply masks
        mask = mask * logits_mask
        if mask_unsup is not None:
            mask_unsup = mask_unsup * logits_mask

        if mask_min is not None:
            mask_maj = mask_maj * logits_mask
            mask_min = mask_min * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-8)

        # Scale by temperature if base_temperature is set
        if self.base_temperature > 0.0:
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        else:
            loss = -1.0 * mean_log_prob_pos

        # Average across all anchors
        loss = loss.view(anchor_count, batch_size).mean()

        # Prepare labels for computing metrics
        batch_labels = torch.arange(batch_size, device=device, dtype=torch.long)
        batch_labels = batch_labels + batch_size - 1  # Remove sim to self
        batch_labels = torch.cat(
            [batch_labels, torch.arange(batch_size, device=device, dtype=torch.long)],
            dim=0,
        )

        # Remove self-similarity from logits
        clean_logits = exp_logits[~torch.eye(batch_size * anchor_count).bool()].view(
            batch_size * anchor_count, -1
        )

        return loss, clean_logits, batch_labels

    def _create_numerator_mask(
        self,
        labels: torch.Tensor,
        ratio: float,
        minority_class: int,
        majority_class: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Create a mask for the numerator that includes all minority class samples
        and a controlled ratio of majority class samples.

        Args:
            labels: Class labels for each sample.
            ratio: Fraction of majority class samples whose positive pairs (with other majority
                   class samples) should be included in the mask. Must be between 0.0 and 1.0.
                   0.0 means no majority-majority positive pairs.
                   1.0 means all potential majority-majority positive pairs are included.
            minority_class: Index of the minority class.
            majority_class: Index of the majority class.

        Returns:
            Tuple of (mask, effective_ratio)
        """
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Ratio must be between 0 and 1, but is {ratio}")

        # Get indices for each class
        minority_indices = torch.where(labels == minority_class)[0]
        majority_indices = torch.where(labels == majority_class)[0]

        # Select subset of majority class samples
        selected_majority_indices = torch.randperm(majority_indices.size(0))[
            : int(ratio * majority_indices.size(0))
        ]
        selected_majority_indices = majority_indices[selected_majority_indices]

        # Combine minority and selected majority indices
        combined_indices = torch.cat([minority_indices, selected_majority_indices])

        # Create mask for selected samples
        selected_labels = labels[combined_indices].view(-1, 1)
        mask_partial = (selected_labels == selected_labels.T).float()

        # Final mask of size labels.size(0) x labels.size(0)
        mask = torch.zeros(labels.size(0), labels.size(0), dtype=torch.float32).to(
            device
        )
        mask[combined_indices.reshape(-1, 1), combined_indices] = mask_partial

        # Add diagonal (self-pairs)
        mask += torch.eye(mask.size(0)).to(device)
        mask.clamp_(max=1)

        return mask, ratio


class ConSupPrototypeLossDDP(nn.Module):
    """
    Supervised contrastive loss function with prototypes for distributed data parallel (DDP) training.

    This implementation extends supervised contrastive learning with prototypes and
    is optimized for distributed training.

    Args:
        temperature: Temperature scaling parameter for the logits
        contrast_mode: Mode for contrasting ("ALL_VIEWS" uses all views as anchors, "ONE_VIEW" uses only first view)
        base_temperature: Base temperature for scaling
        negatives_weight: Weight for negative samples
        eps: Default margin parameter for prototype distance
        eps_0: Class-specific margin for class 0
        eps_1: Class-specific margin for class 1
        minority_cls: Index of the minority class
        max_epoch: Maximum number of training epochs
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "ALL_VIEWS",
        base_temperature: float = 0.07,
        negatives_weight: float = 1.0,
        eps: float = 0.1,
        eps_0: Optional[float] = None,
        eps_1: Optional[float] = None,
        minority_cls: Optional[int] = None,
        max_epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.eps = eps
        self.eps_0 = eps_0
        self.eps_1 = eps_1

        self.prototypes = None
        self.negatives_weight = negatives_weight

        if eps_0 is not None and eps_0 == eps_1:
            self.eps = self.eps_0 = self.eps_1 = eps_0

        self.minority_cls = minority_cls
        self.max_epoch = max_epoch

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the class prototypes to use in the loss calculation."""
        self.prototypes = prototypes

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Supervised Contrastive with Prototypes loss for DDP.

        Args:
            features: Tensor of shape [bsz, n_views, ...] containing the embeddings
                     of the samples and their augmentations.
            labels: Tensor of shape [bsz, 2] where each row is a one-hot encoding
                   of the class label.

        Returns:
            Tuple of (loss, logits, batch_labels)
        """
        # Validate input dimensions
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device

        # Gather features and labels from all devices if using DDP
        if dist.is_available() and dist.is_initialized():
            gathered_features = dist_f.all_gather(features)
            features = torch.cat(gathered_features, dim=0)

            if labels is not None:
                gathered_labels = dist_f.all_gather(labels)
                labels = torch.cat(gathered_labels, dim=0)

        batch_size = features.shape[0]

        num_prototypes = 0
        if self.prototypes is not None:
            num_prototypes = self.prototypes.size(0)

            if num_prototypes != 2:
                raise ValueError("Number of prototypes must be 2")
            if self.prototypes.device != device:
                self.prototypes = self.prototypes.to(device)

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask_maj = mask_min = mask_sup = None

        if labels is not None:
            labels_dim = labels[:, 1].contiguous().view(-1, 1)
            if labels_dim.shape != (batch_size, 1) or labels.shape != (batch_size, 2):
                raise ValueError("Number of labels does not match number of features")

            # Create symmetric mask based on label equality, indicating positive pairs
            mask_sup = torch.eq(labels_dim, labels_dim.T).float().to(device)
            labels_dim = labels_dim.squeeze(1)

            # Create boolean indices for each class
            indices_0 = (labels_dim == 0).to(device)
            indices_1 = (labels_dim == 1).to(device)

            # Initialize class-specific masks
            mask_0, mask_1 = mask_sup.clone(), mask_sup.clone()

            # Set opposite class indices to 0 in each mask
            mask_0[indices_1] = 0
            mask_0[:, indices_1] = 0
            mask_1[indices_0] = 0
            mask_1[:, indices_0] = 0

            # Validate the 'minority_cls' attribute
            if self.minority_cls is None:
                raise ValueError("minority_cls must be set")

            # Assign masks based on minority class identifier
            mask_min, mask_maj = (
                (mask_0, mask_1) if self.minority_cls == 0 else (mask_1, mask_0)
            )

        contrast_count = features.shape[1]  # num of augmentations

        # stack all the augmentations on top of each other
        # first features 0 to bsz and then their augmentations
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.prototypes is not None:
            contrast_feature = torch.cat([contrast_feature, self.prototypes], dim=0)

        if self.contrast_mode == "ONE_VIEW":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "ALL_VIEWS":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # Calculate similarity matrix
        sims = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(sims, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat mask for all views
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,  # value to fill
        )

        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)
            mask_maj = mask_maj * logits_mask

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
            mask_min = mask_min * logits_mask

        if mask_sup is not None:
            mask_sup = mask_sup.repeat(anchor_count, contrast_count)
            mask_sup = mask_sup * logits_mask

        mask = mask * logits_mask

        if self.prototypes is not None:
            if labels is None:
                raise ValueError("labels must be provided if prototypes are provided")

            bsz = labels.shape[0]

            # labels is of shape (bsz, 2,)
            selected_prototypes = labels
            selected_prototypes_mask = selected_prototypes.to(device)

            assert selected_prototypes_mask.shape[0] == bsz
            assert selected_prototypes_mask.shape[1] == num_prototypes

            selected_prototypes_mask = torch.cat(
                [selected_prototypes_mask, selected_prototypes_mask], dim=0
            )  # for the second view

            logits_mask = torch.cat(
                [logits_mask, torch.zeros_like(selected_prototypes_mask).to(device)],
                dim=1,
            )

            logits = logits[:-num_prototypes, :]

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            self.negatives_weight * exp_logits.sum(1, keepdim=True)
        )

        if self.prototypes is not None:
            m_pull = torch.ones_like(selected_prototypes_mask, dtype=torch.float32).to(
                device
            )

            # No pull for samples that are closer to their own prototype than the other prototype
            cond2 = sims[:-num_prototypes, -1] <= (
                sims[:-num_prototypes, -2] + self.eps_1
            )
            cond1 = sims[:-num_prototypes, -2] <= (
                sims[:-num_prototypes, -1] + self.eps_0
            )

            # Apply conditions and selected_prototypes_mask together
            p2_mask = torch.logical_and(cond2, selected_prototypes_mask[:, -1].bool())
            p1_mask = torch.logical_and(cond1, selected_prototypes_mask[:, -2].bool())

            selected_prototypes_mask = torch.stack((p1_mask, p2_mask), dim=1).float()

            m_pull = m_pull * selected_prototypes_mask
            mask = torch.cat([mask, m_pull], dim=1)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        if self.base_temperature > 0.0:
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        else:
            loss = -1.0 * mean_log_prob_pos

        loss = loss.mean()

        # Prepare labels for metrics
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = labels + batch_size - 1  # Remove sim to self
        labels = torch.cat(
            [labels, torch.arange(batch_size, device=device, dtype=torch.long)], dim=0
        )

        if self.prototypes is not None:
            exp_logits = exp_logits[:, :-num_prototypes]

        # Remove self-similarity from logits
        clean_logits = exp_logits[~torch.eye(batch_size * anchor_count).bool()].view(
            batch_size * anchor_count, -1
        )

        return loss, clean_logits.float().detach(), labels.detach()
