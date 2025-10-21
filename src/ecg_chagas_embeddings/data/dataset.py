from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from ecg_chagas_embeddings.data.augmentation import ECGAugmentation
from ecg_chagas_embeddings.data.prepare_dataset import (
    preprocess_ecg_safe,
    softclip_scale_ecg,
)


def read_metadata_folds(
    meta_path: Path, folds: Optional[Union[int, Iterable]] = None
) -> pd.DataFrame:
    metadata = pd.read_csv(meta_path, low_memory=False)
    if folds:
        folds = folds if isinstance(folds, (list, tuple)) else [folds]
        min_fold = metadata.fold.min()
        max_fold = metadata.fold.max()
        assert max(folds) <= max_fold and min(folds) >= min_fold, (
            f"Allowed folds are between {min_fold} and {max_fold}"
        )
        metadata = metadata[metadata.fold.isin(folds)]
    return metadata


# def get_sampling_weights(y: pd.Series) -> torch.DoubleTensor:
#     """Sampling weights for balanced sampling."""
#     label_to_count = y.value_counts()
#     weights = 1.0 / label_to_count[y]
#     return torch.DoubleTensor(weights.to_list())


def get_sampling_weights(
    y: pd.Series,
    sources: pd.Series,
    pos_weights_per_source: Optional[dict] = None,
    neg_weights_per_source: Optional[dict] = None,
) -> torch.DoubleTensor:
    """
    Compute per-sample weights based on inverse frequency and custom source-based class weightings.
    """
    df = pd.DataFrame({"label": y, "source": sources})

    # Compute group frequency
    group_counts = df.groupby(["source", "label"]).size()
    group_weights = 1.0 / group_counts

    print("DEBUG: Group counts:\n", group_counts)
    print("DEBUG: Group weights (inverse frequency):\n", group_weights)

    # Multiply group counts by group weights and print
    product = group_counts * group_weights
    print("DEBUG: Group counts * group weights:\n", product)

    # Build per-sample base weights
    weights = df.apply(
        lambda row: group_weights.get((row["source"], row["label"]), 0.0), axis=1
    )

    # print the unique weights for debugging
    print("DEBUG: Unique weights before applying custom multipliers:", weights.unique())

    # Apply custom multipliers (optional)
    for src, w in (pos_weights_per_source or {}).items():
        if w == 0.0:
            weights[(df["source"] == src) & (df["label"] == 1)] = w
        else:
            weights[(df["source"] == src) & (df["label"] == 1)] *= w
        print(
            f"DEBUG: Applying positive weight {w} for source {src} -> ",
            weights[(df["source"] == src) & (df["label"] == 1)].unique(),
        )
    for src, w in (neg_weights_per_source or {}).items():
        if w == 0.0:
            weights[(df["source"] == src) & (df["label"] == 0)] = w
        else:
            weights[(df["source"] == src) & (df["label"] == 0)] *= w
        print(
            f"DEBUG: Applying negative weight {w} for source {src} -> ",
            weights[(df["source"] == src) & (df["label"] == 0)].unique(),
        )

    # Normalize weights to ensure they sum to 1
    weights /= weights.sum()

    return torch.DoubleTensor(weights.values)


def get_sampling_weights_pos_frac(
    y: pd.Series,
    sources: pd.Series,
    pos_weights_per_source: Optional[dict] = None,
    neg_weights_per_source: Optional[dict] = None,
    desired_positive_ratio: float = 0.5,
) -> torch.DoubleTensor:
    """
    Compute per-sample weights based on inverse frequency, custom source-based multipliers,
    and optionally rebalance to a target positive class ratio (e.g., 0.5 for 50/50).
    """
    df = pd.DataFrame({"label": y, "source": sources})

    # Compute group frequency
    group_counts = df.groupby(["source", "label"]).size()
    group_weights = 1.0 / group_counts

    print("DEBUG: Group counts:\n", group_counts)
    print("DEBUG: Group weights (inverse frequency):\n", group_weights)

    weights = df.apply(
        lambda row: group_weights.get((row["source"], row["label"]), 0.0), axis=1
    )

    print("DEBUG: Unique weights before custom multipliers:", weights.unique())

    # Apply optional source-specific class weights
    for src, w in (pos_weights_per_source or {}).items():
        weights[(df["source"] == src) & (df["label"] == 1)] *= w
    for src, w in (neg_weights_per_source or {}).items():
        weights[(df["source"] == src) & (df["label"] == 0)] *= w

    # Apply global positive/negative rescaling to match desired ratio
    pos_mask = df["label"] == 1
    neg_mask = df["label"] == 0
    pos_sum = weights[pos_mask].sum()
    neg_sum = weights[neg_mask].sum()
    total = pos_sum + neg_sum

    if pos_sum > 0 and neg_sum > 0:
        desired_pos_sum = total * desired_positive_ratio
        desired_neg_sum = total * (1 - desired_positive_ratio)
        weights[pos_mask] *= desired_pos_sum / pos_sum
        weights[neg_mask] *= desired_neg_sum / neg_sum

        print(
            f"DEBUG: Rebalanced to target positive ratio: {desired_positive_ratio:.2f}"
        )
        print(
            f"DEBUG: New pos/neg sums: {weights[pos_mask].sum():.4f} / {weights[neg_mask].sum():.4f}"
        )

    # Normalize final weights
    weights /= weights.sum()

    return torch.DoubleTensor(weights.values)


def collate_dict_batch(batch):
    # print("DEBUG BATCH[0]:", batch[0])

    # batch is a list of dicts
    batch_out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            batch_out[key] = torch.stack([d[key] for d in batch])
        else:
            batch_out[key] = [d[key] for d in batch]  # e.g., exam_id (string)
    return batch_out


class WfdbDataset(Dataset):
    def __init__(
        self,
        meta_path: Path,
        transforms: Optional[List[Callable]] = None,
        folds: Optional[Union[int, Iterable]] = None,
    ):
        """
        Args:
            meta_path: Path to the metadata file
            transforms: Transforms to apply to the ecg.
            folds: List of folds to use.
        """
        self.metadata = read_metadata_folds(meta_path, folds)
        self.metadata = self.metadata[~self.metadata.chagas.isna()]
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, float]:
        row = self.metadata.iloc[index]

        record = wfdb.rdrecord(row.path)

        ecg = record.p_signal.T  # Transpose to (n_channels, n_samples)
        label = float(row.chagas)

        if self.transforms:
            ecg = self.transforms(ecg)

        return ecg, label


class TorchDataset(Dataset):
    def __init__(
        self,
        meta_path: Path,
        data_dir: Path,
        transforms: Optional[Callable] = None,
        use_sup_con_views: Optional[
            int
        ] = None,  # optional; mostly handled by transform's n_views
        folds: Optional[Union[int, Iterable]] = None,
        return_age_and_sex: bool = False,
        use_code15: bool = True,
        use_ptb_xl: bool = True,
        use_sami_trop: bool = True,
        is_submission: bool = False,
    ):
        self.metadata = read_metadata_folds(meta_path, folds)
        if not use_sami_trop:
            self.metadata = self.metadata[self.metadata.source != "SaMi-Trop"]
        if not use_ptb_xl:
            self.metadata = self.metadata[self.metadata.source != "PTB-XL"]
        if not use_code15:
            self.metadata = self.metadata[self.metadata.source != "CODE-15%"]
        self.metadata = self.metadata[~self.metadata.chagas.isna()]
        self.metadata.drop_duplicates(subset="exam_id", ignore_index=True, inplace=True)

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.use_sup_con_views = use_sup_con_views
        self.is_submission = is_submission

        self.return_age_and_sex = return_age_and_sex
        if return_age_and_sex:
            self.metadata["age"] = self.metadata["age"].clip(0, 100) / 100.0
            d = {"Male": 0.0, "Female": 1.0}
            self.metadata["sex"] = self.metadata["sex"].map(lambda x: d.get(x, x))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]

        if self.is_submission:
            # training from wfdb
            signal, _ = preprocess_ecg_safe(
                row["path"], target_sample_rate=400
            )  # (C, L) numpy

            # Apply soft clipping using dataset statistics

            global_lower = np.array(
                [
                    -2.3736463,
                    -2.685248,
                    -2.200961,
                    -2.169291,
                    -2.0341082,
                    -2.218963,
                    -3.18454,
                    -3.967165,
                    -4.248691,
                    -4.4520607,
                    -4.58838,
                    -4.0192003,
                ],
                dtype=np.float32,
            )

            global_upper = np.array(
                [
                    2.3736463,
                    2.685248,
                    2.200961,
                    2.169291,
                    2.0341082,
                    2.218963,
                    3.18454,
                    3.967165,
                    4.248691,
                    4.4520607,
                    4.58838,
                    4.0192003,
                ],
                dtype=np.float32,
            )

            # If your training used "corner sharpness" c equal to the per-lead 99th (same as upper),
            # keep it identical. If you had a separate c-array, use that instead.
            softclip_c = global_upper.copy()

            # Use your exact training function to avoid formula drift:
            signal = softclip_scale_ecg(
                signal, a=global_lower, b=global_upper, c=softclip_c
            )
            ecg = torch.tensor(signal, dtype=torch.float32)  # (1, C, L)
        else:
            # training from tensors
            ecg = torch.load(self.data_dir / f"{row.exam_id}.pt").to(torch.float32)
        chagas = torch.tensor([row.chagas], dtype=torch.float32)

        # source -> index (if you need it)
        source_map = {"CODE-15%": 0, "PTB-XL": 1, "SaMi-Trop": 2}
        source_idx = source_map.get(row.source, -1)

        # --- apply transform ONCE ---
        if self.transforms is not None:
            out = self.transforms(ecg)  # returns [C,T] or [V,C,T] based on n_views
        else:
            out = ecg

        # If someone forgot to set n_views in the transform but passed use_sup_con_views,
        # make a best-effort stack here:
        if (self.use_sup_con_views and self.use_sup_con_views > 1) and out.dim() == 2:
            out = out.unsqueeze(0).repeat(self.use_sup_con_views, 1, 1)  # [V,C,T]

        # --- build sample ---
        if not self.return_age_and_sex:
            # keep backward-compatible tuple return
            return out, chagas

        # One-hot-ish sex value (keeps your original mapping logic)
        sex_value = 1.0 if getattr(row, "sex", "Female") == 0.0 else 0.0

        sample = {
            "chagas": chagas,
            "age": torch.tensor([getattr(row, "age", 0.0)], dtype=torch.float32),
            "sex": torch.tensor([sex_value], dtype=torch.float32),
            "source": torch.tensor([source_idx], dtype=torch.float32),
            "exam_id": getattr(row, "exam_id", ""),
        }

        # put under the right key
        if out.dim() == 3:  # [V,C,T] -> training/SupCon
            sample["ecg_views"] = out
        else:  # [C,T]   -> validation/inference
            sample["ecg"] = out

        return sample


def get_train_val_loaders(
    meta_path: Path,
    data_dir: Path,
    batch_size: int,
    oversample: bool = False,
    pos_fraction: float = 0.05,
    train_folds: Union[int, Iterable] = [0, 1, 2, 3],
    valid_folds: Union[int, Iterable] = [4],
    num_workers: int = 0,
    prefetch_factor: int = 2,
    neg_weight_code15: float = 1.0,
    neg_weight_ptb_xl: float = 1.0,
    neg_weight_sami_trop: float = 1.0,
    pos_weight_code15: float = 1.0,
    pos_weight_ptb_xl: float = 1.0,
    pos_weight_sami_trop: float = 1.0,
    *,
    crop_size: int = 2500,
    max_time_warp: float = 0.2,
    scaling: Tuple[float, float] = (0.8, 1.2),
    gaussian_noise_std: float = 0.01,
    wandering_max_amplitude: float = 1.0,
    wandering_frequency_range: Tuple[float, float] = (0.5, 2.0),
    max_mask_duration: int = 50,
    mask_prob: float = 0.5,
    is_submission=False,
    use_sup_con=False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for training and validation. Keyword arguments are used to configure the
    ECGAugmentation for the training dataset.

    Args:
        meta_path (Path): Path to the metadata file containing dataset information.
        data_dir (Path): Path to the directory where the dataset is stored.
        batch_size (int): Number of samples per batch.
        oversample (bool, optional): Whether to apply oversampling for balancing classes. Defaults to False.
        train_folds (Union[int, Iterable], optional): Fold indices to use for training. Defaults to [0, 1, 2, 3].
        valid_folds (Union[int, Iterable], optional): Fold indices to use for validation. Defaults to [4].
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 0.
        prefetch_factor (int, optional): Number of samples to prefetch. Defaults to 2.
        crop_size: Crops or pads to this size.
        max_time_warp: Warps time by this percentage.
        scaling: Min and max of amplitude scaling.
        gaussian_noise_std: Gaussian noise std.
        wandering_max_amplitude: Amplitude of random wandering.
        wandering_frequency_range: Frequency range of random wandering.
        max_mask_duration: Max duration of zero masking.
        mask_prob: Probability to completely mask a lead (channel).

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        A tuple containing the training and validation DataLoaders.
    """

    train_transform_kwargs = {
        "crop_size": crop_size,
        "n_views": 2 if use_sup_con else 1,  # << two views for SupCon
    }

    # (shared, light) masks
    if max_mask_duration and max_mask_duration > 0:
        train_transform_kwargs["max_mask_duration"] = max_mask_duration
    if mask_prob and mask_prob > 0:
        train_transform_kwargs["mask_prob"] = mask_prob

    # (per-view) appearance tweaks
    if gaussian_noise_std and gaussian_noise_std > 0:
        train_transform_kwargs["gaussian_noise_std"] = gaussian_noise_std
        train_transform_kwargs["per_view_noise"] = True  # default, explicit here

    if scaling and (scaling[0] != 1.0 or scaling[1] != 1.0):
        train_transform_kwargs["scaling"] = scaling
        train_transform_kwargs["per_view_scaling"] = True  # default, explicit here

    # (tiny, usually OFF) warp — if you enable it, keep same warp across views
    if max_time_warp and max_time_warp > 0:
        train_transform_kwargs["max_time_warp"] = max_time_warp
        train_transform_kwargs["per_view_warp"] = False

    # (usually OFF if you bandpass at train+infer) wandering
    if (
        wandering_max_amplitude
        and wandering_max_amplitude > 0
        and wandering_frequency_range
        and (wandering_frequency_range[0] > 0 or wandering_frequency_range[1] > 0)
    ):
        train_transform_kwargs["wandering_max_amplitude"] = wandering_max_amplitude
        train_transform_kwargs["wandering_frequency_range"] = wandering_frequency_range
        train_transform_kwargs["per_view_wandering"] = False

    print("Train Transform kwargs:", train_transform_kwargs)
    train_transform = ECGAugmentation(
        crop_size=train_transform_kwargs.get("crop_size", crop_size),
        n_views=train_transform_kwargs.get("n_views", 2 if use_sup_con else 1),
        max_mask_duration=train_transform_kwargs.get("max_mask_duration", None),
        mask_prob=train_transform_kwargs.get("mask_prob", None),
        gaussian_noise_std=train_transform_kwargs.get("gaussian_noise_std", None),
        per_view_noise=bool(train_transform_kwargs.get("per_view_noise", True)),
        scaling=cast(
            Optional[Tuple[float, float]], train_transform_kwargs.get("scaling", None)
        ),
        per_view_scaling=bool(train_transform_kwargs.get("per_view_scaling", True)),
        max_time_warp=train_transform_kwargs.get("max_time_warp", None),
        per_view_warp=bool(train_transform_kwargs.get("per_view_warp", False)),
        wandering_max_amplitude=train_transform_kwargs.get(
            "wandering_max_amplitude", None
        ),
        wandering_frequency_range=cast(
            Optional[Tuple[float, float]],
            train_transform_kwargs.get("wandering_frequency_range", None),
        ),
        per_view_wandering=bool(
            train_transform_kwargs.get("per_view_wandering", False)
        ),
    )

    # --- VALIDATION: single, clean view ---
    valid_transform = ECGAugmentation(
        crop_size=crop_size,
        n_views=1,  # << single view
        # Keep val clean/deterministic; typically no masks/noise/warp here.
    )
    train_dataset = TorchDataset(
        meta_path,
        data_dir,
        transforms=train_transform,
        # transforms=ECGAugmentation(crop_size=crop_size),
        folds=train_folds,
        return_age_and_sex=True,
        use_code15=pos_weight_code15 > 0 or neg_weight_code15 > 0,
        use_ptb_xl=pos_weight_ptb_xl > 0 or neg_weight_ptb_xl > 0,
        use_sami_trop=pos_weight_sami_trop > 0 or neg_weight_sami_trop > 0,
        is_submission=is_submission,
    )
    valid_dataset = TorchDataset(
        meta_path,
        data_dir,
        transforms=valid_transform,
        folds=valid_folds,
        return_age_and_sex=True,
        use_code15=pos_weight_code15 > 0 or neg_weight_code15 > 0,
        use_ptb_xl=pos_weight_ptb_xl > 0 or neg_weight_ptb_xl > 0,
        use_sami_trop=pos_weight_sami_trop > 0 or neg_weight_sami_trop > 0,
        is_submission=is_submission,
    )

    if oversample:
        weights = get_sampling_weights_pos_frac(
            train_dataset.metadata.chagas,
            sources=train_dataset.metadata.source,
            pos_weights_per_source={
                "CODE-15%": pos_weight_code15,
                "PTB-XL": pos_weight_ptb_xl,
                "SaMi-Trop": pos_weight_sami_trop,
            },
            neg_weights_per_source={
                "CODE-15%": neg_weight_code15,
                "PTB-XL": neg_weight_ptb_xl,
                "SaMi-Trop": neg_weight_sami_trop,
            },
            desired_positive_ratio=pos_fraction,
        )
        # WeightedRandomSampler expects a sequence of floats (e.g., list); convert tensor to list
        sampler = WeightedRandomSampler(
            weights.tolist(), len(train_dataset), replacement=True
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
        shuffle=sampler is None,
        sampler=sampler,
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        collate_fn=collate_dict_batch,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_dict_batch,
    )

    return train_loader, valid_loader
