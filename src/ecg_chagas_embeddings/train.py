#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    # StochasticWeightAveraging,
    ModelCheckpoint,
    LearningRateMonitor,
    # LearningRateFinder,
)
import lightning as pl
import yaml

# from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from ecg_chagas_embeddings.data.dataset import get_train_val_loaders
import glob
import re
from ecg_chagas_embeddings.models.resnet18_ecg_flex import BasicBlock, LitResNet18NJ
from ecg_chagas_embeddings.data.prepare_dataset import (
    preprocess_ecg_safe,
    softclip_scale_ecg,
)
from ecg_chagas_embeddings.data.augmentation import RandomCropOrPad

# trade off a bit of precision for substantially higher throughput
# torch.set_float32_matmul_precision('medium')
# or even
torch.set_float32_matmul_precision("high")


disable_tqdm = not sys.stdout.isatty()

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.


@dataclass
class Config:
    local_data_folder: Path = Path("/sc-scratch/sc-scratch-dh-face/physionet2025/")
    model_folder: Path = Path("model")
    pre_model_folder: Path = Path("models/nj_resnet_init/model.pth")
    data_subdir: str = "processedOfficialCleanV3"
    meta_path: str = "metadata_official_clean_v3.csv"
    allow_failures: bool = False
    # Optimizer specific parameters
    optimizer: str = "adamw"
    momentum: float = 0.9
    lr: float = 0.0005
    lr_scheduler: str = "one_cycle"  # options: none, one_cycle, cosine, step
    classifier_weight_decay: float = 1e-5
    params_weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    se_reduction: int = 0  # 0 means no SE block, >0 means SE block with reduction ratio
    crop_size: int = 2500
    max_time_warp: float = 0.15
    scaling_diff: float = 0.0
    gaussian_noise_std: float = 0.0
    mask_prob: float = 0.1
    max_mask_duration: int = 100
    wandering_max_amplitude: float = 0.0
    wandering_freq_range: tuple = (0.0, 0.0)
    swa_start: int = 75
    swa_lr: float = 0.01
    num_epochs: int = 25
    pos_fraction: float = 0.25
    norm_type: str = "group"  # "none","batch", "group", "instance", "layer"
    norm_groups: int = 8
    # Loss specific parameters
    criterion: str = "focal"
    focal_alpha: float = 0.9
    focal_gamma: float = 1.5
    focal_end_gamma: float = 3.0
    asl_gamma_neg: int = 2
    asl_gamma_pos: int = 1
    asl_clip: float = 0.01
    tversky_alpha: float = 0.5
    tversky_beta: float = 0.5
    tversky_smooth: float = 1.0
    focal_tversky_alpha: float = 0.6
    focal_tversky_beta: float = 0.4
    focal_tversky_gamma: float = 1.0
    focal_tversky_smooth: float = 0.1
    bce_pos_weight: float = 1.0
    classifier_weight: float = 1.0
    use_sup_con: bool = False
    use_prototypes: bool = False
    sup_con_weight: float = 0.05
    sup_con_temp: float = 0.07
    # Dataloader specific parameters
    batch_size: int = 256
    oversample: bool = True
    neg_weight_code15: float = 0.5
    neg_weight_ptb_xl: float = 0.5
    neg_weight_sami_trop: float = 1.0
    pos_weight_code15: float = 0.5
    pos_weight_ptb_xl: float = 1.0
    pos_weight_sami_trop: float = 0.5
    loss_weight_code15: float = 1.0
    loss_weight_ptb_xl: float = 1.0
    loss_weight_sami_trop: float = 1.0
    train_folds: tuple = (0, 1, 2, 3)
    valid_folds: tuple = (4,)
    num_workers: int = 4
    prefetch_factor: int = 16


def save_config(config: Config, filepath: Path):
    config_dict = asdict(config)
    # Convert Path objects to strings for YAML compatibility
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)
        # Convert tuples to lists for YAML compatibility
        if isinstance(v, tuple):
            config_dict[k] = list(v)

    with open(filepath, "w") as f:
        yaml.dump(config_dict, f)


def load_config(filepath: Path) -> Config:
    with open(filepath, "r") as f:
        config_dict = yaml.safe_load(f)
    # Convert string paths back to Path objects
    path_fields = [
        "local_data_folder",
        "model_folder",
        "pre_model_folder",
    ]  # Add others if needed
    for field in path_fields:
        if field in config_dict:
            config_dict[field] = Path(config_dict[field])
    # Convert lists back to tuples
    tuple_fields = ["train_folds", "valid_folds"]  # Add other tuple fields if needed
    for field in tuple_fields:
        if field in config_dict and isinstance(config_dict[field], list):
            config_dict[field] = tuple(config_dict[field])
    for field in path_fields:
        if field in config_dict:
            config_dict[field] = Path(config_dict[field])

    return Config(**config_dict)


def get_criterion(config: Config):
    if config.criterion == "focal":
        return FocalLoss(
            alpha=config.focal_alpha, gamma=config.focal_gamma, reduction="mean"
        )
    if config.criterion == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
        )
    if config.criterion == "tversky":
        return TverskyLoss(
            alpha=config.tversky_alpha,
            beta=config.tversky_beta,
            smooth=config.tversky_smooth,
        )
    if config.criterion == "focal_tversky":
        return FocalTverskyLoss(
            alpha=config.focal_tversky_alpha,
            beta=config.focal_tversky_beta,
            gamma=config.focal_tversky_gamma,
            smooth=config.focal_tversky_smooth,
            k=2 * config.pos_fraction,
            tau=0.5,
            entropy_weight=0.005,
            bce_weight=0.1,
            pos_weight=(1.0 - config.pos_fraction) * 10,
        )
    if config.criterion == "bce":
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor(config.bce_pos_weight)
        )
    if config.criterion == "source_weighted_bce":
        source_weights = {
            0: config.loss_weight_code15,
            1: config.loss_weight_ptb_xl,
            2: config.loss_weight_sami_trop,
        }
        return SourceWeightedBCE(source_weights=source_weights, reduction="mean")
    if config.criterion == "weighted_top_tversky":
        source_weights = {
            0: config.loss_weight_code15,
            1: config.loss_weight_ptb_xl,
            2: config.loss_weight_sami_trop,
        }
        return SourceWeightedTopTverskyLoss(
            source_weights=source_weights,
            alpha=0.3,
            beta=0.7,
            top_percent=0.05,
            tau=0.1,
            use_bce=True,
            bce_weight=0.1,
            use_focal=True,
            focal_weight=0.2,
            focal_gamma=2.0,
            reduction="mean",
        )

    if config.criterion == "ranking_bce":
        return RankingBCELoss()

    raise ValueError(f"Unknown criterion: {config.criterion}")


def check_if_submission(data_folder, config) -> bool:
    if data_folder == config.local_data_folder:
        print("This is not a submission!")
        return False
    return True


def prepare_dataloaders(
    data_folder: Path, model_folder: Path, config: Config, verbose: bool = True
):
    if verbose:
        print("Finding the Challenge data...")

    is_submission = check_if_submission(data_folder, config)

    if not os.path.exists(
        os.path.join(model_folder if is_submission else data_folder, "metadata.csv")
    ):
        if verbose:
            print("Creating metadata...")
        os.system(
            f"python prepare_dataset.py --processes 0 --data_dir {data_folder} --save_meta_only --output_dir {model_folder if is_submission else data_folder}/processed --output_file metadata.csv"
        )

    if is_submission:
        meta_path = model_folder / "processed" / "metadata.csv"
        data_dir = data_folder
    else:
        meta_path = data_folder / config.meta_path
        data_dir = data_folder / config.data_subdir

    train_dataloader, valid_dataloader = get_train_val_loaders(
        meta_path,
        data_dir,
        batch_size=config.batch_size,
        oversample=config.oversample,
        pos_fraction=config.pos_fraction,
        train_folds=config.train_folds,
        valid_folds=config.valid_folds,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        crop_size=config.crop_size,
        max_time_warp=config.max_time_warp,
        scaling=(1 - config.scaling_diff, 1 + config.scaling_diff),
        gaussian_noise_std=config.gaussian_noise_std,
        mask_prob=config.mask_prob,
        max_mask_duration=config.max_mask_duration,
        wandering_max_amplitude=config.wandering_max_amplitude,
        wandering_frequency_range=config.wandering_freq_range,
        neg_weight_code15=config.neg_weight_code15,
        neg_weight_ptb_xl=config.neg_weight_ptb_xl,
        neg_weight_sami_trop=config.neg_weight_sami_trop,
        pos_weight_code15=config.pos_weight_code15,
        pos_weight_ptb_xl=config.pos_weight_ptb_xl,
        pos_weight_sami_trop=config.pos_weight_sami_trop,
        is_submission=is_submission,
        use_sup_con=config.use_sup_con or config.use_prototypes,
    )
    return train_dataloader, valid_dataloader


def summarize_weights(model: nn.Module) -> dict:
    """Return histogram statistics of the classifier weights, including quartiles."""
    if not hasattr(model, "fc") or not hasattr(model.fc, "weight"):
        raise AttributeError("The model has no attribute 'fc.weight'.")

    w = model.fc.weight

    # Handle different possible types
    if isinstance(w, torch.Tensor):
        weights = w.detach().cpu().numpy().flatten()
    elif isinstance(w, nn.Parameter):
        weights = w.data.cpu().numpy().flatten()
    elif isinstance(w, (np.ndarray, list)):
        weights = np.array(w).flatten()
    else:
        raise TypeError(f"Unsupported weight type: {type(w)}")

    return {
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "q1": float(np.percentile(weights, 25)),
        "median": float(np.percentile(weights, 50)),
        "q3": float(np.percentile(weights, 75)),
    }


def init_linear_layer_weights(
    model, init_method="random", std=0.2, scale=5, verbose=True
):
    """
    Initialize the weights of the model's linear layer using the specified method.
    Args:
        model (torch.nn.Module): The model whose weights are to be initialized.
        init_method (str): The initialization method to use. Options are "random", "xavier", "kaiming", or "scale".
        std (float): Standard deviation for random initialization.
        scale (float): Scaling factor for scaling weights using "scale" init_method.
        verbose (bool): If True, print the weight histogram statistics before and after normalization.
    Returns:
        model (torch.nn.Module): The model with initialized weights.
    """

    if verbose:
        _summary = summarize_weights(model)
        print("Weight Histogram Stats - Before Normalization")
        for key, value in _summary.items():
            print(f"\t{key}: {value:.4f}")

    if init_method == "random":
        torch.nn.init.normal_(model.fc.weight, mean=0.0, std=std)
    elif init_method == "xavier":
        torch.nn.init.xavier_uniform_(model.fc.weight)
    elif init_method == "kaiming":
        torch.nn.init.kaiming_uniform_(model.fc.weight, nonlinearity="relu")
    elif init_method == "scale":
        model.fc.weight.data *= scale
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")

    if model.fc.bias is not None:
        if verbose:
            print("Bias is not None, setting to 0")
        torch.nn.init.constant_(model.fc.bias, 0.0)

    if verbose:
        _summary = summarize_weights(model)
        print("Weight Histogram Stats - After Normalization")
        for key, value in _summary.items():
            print(f"\t{key}: {value:.4f}")

    return model


def get_default_args():
    return {
        "data_subdir": "processedOfficialCleanV3",
        "meta_path": "metadata_official_clean_v3.csv",
        "num_epochs": 25,
        "num_workers": 4,
        "batch_size": 256,
        "classifier_weight_decay": 1e-5,  # for classifier weights, e.g. linear layer
        "params_weight_decay": 1e-5,  # for all other parameters,
        "pos_fraction": 0.25,
        "norm_type": "group",  # "none", "batch", "group", "instance", "layer"
        "norm_groups": 8,
        "prefetch_factor": 16,
        "oversample": True,
        "dropout_rate": 0.025,
        "se_reduction": 16,  # 0 means no SE block, >0 means SE block with reduction ratio
        "use_sup_con": False,
        "use_prototypes": False,
        "classifier_weight": 1.0,
        "sup_con_weight": 0.5,
        "sup_con_temp": 0.07,
        "crop_size": 2500,
        "max_time_warp": 0.05,
        "scaling_diff": 0.0,
        "gaussian_noise_std": 0.01,
        "mask_prob": 0.05,
        "max_mask_duration": 100,
        "wandering_max_amplitude": 0.0,  # 1.0
        "wandering_freq_range": (0.0, 0.0),
        "neg_weight_code15": 0.5,
        "neg_weight_ptb_xl": 1.0,
        "neg_weight_sami_trop": 1.0,
        "pos_weight_code15": 0.5,
        "pos_weight_ptb_xl": 1.0,
        "pos_weight_sami_trop": 1.0,
        "loss_weight_code15": 1.0,
        "loss_weight_ptb_xl": 1.0,
        "loss_weight_sami_trop": 1.0,
        "lr": 1e-4,
        "lr_scheduler": "none",  # options: none, one_cycle, cosine, step
        "optimizer": "adamw",
        "criterion": "focal",
        "bce_pos_weight": 1.0,
        "focal_alpha": 0.9,
        "focal_gamma": 1.5,
        "focal_smooth": 1e-8,
        "tversky_alpha": 0.5,
        "tversky_beta": 0.5,
        "tversky_smooth": 1.0,
        "focal_tversky_alpha": 0.6,
        "focal_tversky_beta": 0.4,
        "focal_tversky_gamma": 1.0,
        "focal_tversky_smooth": 0.1,
        "asl_gamma_neg": 4,
        "asl_gamma_pos": 1,
        "experiment": "default_experiment",
        "run_name": "default_run",
        "train_folds": [0, 1, 2, 3],
        "valid_folds": [4],
    }


# Train your model.
def train_model(data_folder, model_folder, verbose, args=None):
    data_folder = Path(data_folder)
    model_folder = Path(model_folder)

    if args is None:
        defaults = get_default_args()
        args = argparse.Namespace(**defaults)
        args.use_sup_con = bool(args.use_sup_con)
        args.use_prototypes = bool(args.use_prototypes)
    else:
        # Ensure folds are parsed if passed as string
        if isinstance(args.train_folds, str):
            args.train_folds = list(map(int, args.train_folds.split(",")))
        if isinstance(args.valid_folds, str):
            args.val_folds = list(map(int, args.valid_folds.split(",")))
        if isinstance(args.wandering_freq_range, str):
            args.wandering_freq_range = tuple(
                map(float, args.wandering_freq_range.strip("()").split(","))
            )

        # ensure that booleans are parsed correctly
        # args.oversample = bool(args.oversample)
        # args.use_sup_con = bool(args.use_sup_con)
        # args.use_prototypes = bool(args.use_prototypes)

        # print boolean arguments
        print(f"Using oversampling: {args.oversample}")
        print(f"Using supervised contrastive loss: {args.use_sup_con}")
        print(f"Using SupCon Prototypes: {args.use_prototypes}")

    # Build config, only set fields if the argument exists in args
    config_kwargs = {}
    for field in Config.__dataclass_fields__:
        if hasattr(args, field):
            config_kwargs[field] = getattr(args, field)
    config = Config(**config_kwargs)

    print("Training with config:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    device = set_device(verbose=verbose)
    pl.seed_everything(42, workers=True)
    # set_seeds(42)

    # Ensure the model_folder exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Find the data files.
    if config.num_epochs == 0:
        print("num_epochs is 0, skipping training.")
        return
    train_dataloader, valid_dataloader = prepare_dataloaders(
        data_folder, model_folder, config, verbose
    )
    if verbose:
        print(
            f"Found {len(train_dataloader)} training batches and {len(valid_dataloader)} validation batches."
        )

    # Save them to a .yaml file in model_folder
    save_config(config, model_folder / "run_config.yaml")

    # model = load_model(model_folder, verbose)

    model = create_model(config)

    # pos_fraction = config.pos_fraction
    # neg_fraction = 1 - pos_fraction
    # bias = np.log(pos_fraction / neg_fraction)
    # lin for old model, fc for NJs variant
    # model.fc.bias.data.fill_(bias)
    # print(f"Bias set to {bias:.4f}")

    model.to(device)

    # save_model(model_folder, model, config)
    # print(f"Model saved to {model_folder}")

    # Train the models.
    if verbose:
        print("Training the model on the data...")

    ckpt = ModelCheckpoint(
        monitor="val/score",
        dirpath=config.model_folder,
        mode="max",
        filename="best_ep_{epoch}_score_{val_score:.3f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_weights_only=False,
    )

    # lr_finder = LearningRateFinder(
    #     min_lr=1e-8,  # lowest LR to try
    #     max_lr=1.1,  # highest LR to try
    #     num_training_steps=100,  # how many steps over which to sweep
    #     mode="exponential",  # or "linear"
    #     early_stop_threshold=None,  # stop if loss spikes >4× best_loss
    #     update_attr=True,  # write the LR back to your model
    #     attr_name="max_lr",  # name of your LR attribute in your LightningModule
    # )

    # swa = StochasticWeightAveraging(
    #     swa_lrs=config.swa_lr,
    #     swa_epoch_start=config.swa_start,
    #     annealing_strategy="cos",
    # )

    # Define experiment name (e.g., "use_swag")
    experiment_name = (
        args.experiment if hasattr(args, "experiment") else "default_experiment"
    )

    # Use timestamp to uniquely identify a run
    run_name = (
        args.run_name
        if hasattr(args, "run_name")
        else f"{experiment_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )

    valid_folds = getattr(args, "valid_folds", 0)

    if isinstance(valid_folds, int):
        fold_id = str(valid_folds)
    elif isinstance(valid_folds, (list, tuple)) and len(valid_folds) > 0:
        fold_id = "+".join(str(f) for f in sorted(valid_folds))
    else:
        fold_id = "0"

    fold_name = f"fold_{fold_id}"

    print(f"Experiment: {experiment_name}, Run: {run_name}, Fold: {fold_name}")

    logger = True  # log to csv when not running on cluster

    if (
        "WANDB_API_KEY" in os.environ
        and not os.environ.get("WANDB_DISABLED", "false").lower() == "true"
    ):
        logger = WandbLogger(
            project="PhysioNetChallenge25",
            group=run_name,
            name=fold_name,
            entity="ag-lukassen",
            save_dir="wandb",
            log_model="all",
            # offline=check_if_submission(data_folder, config),
            config=asdict(config),
        )
    else:
        print("⚠️ W&B logging disabled (no API key or WANDB_DISABLED=true).")

    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     max_steps=100,            # or whatever you want for the sweep
    #     limit_train_batches=0.2,  # optional, speed up by using a fraction
    #     # …no LearningRateFinder here…
    # )

    # # 2) create a Tuner and run the range test
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(
    #     model,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=valid_dataloader,
    #     min_lr=1e-6,
    #     max_lr=1.0,
    #     num_training=100,
    #     early_stop_threshold=None,  # don’t bail out early
    # )

    # # 3) plot & save
    # fig = lr_finder.plot(suggest=True)    # now you get back a matplotlib Figure
    # fig.savefig("lr_finder.png")
    # print("Saved LR plot to:", os.path.abspath("lr_finder.png"))

    # # 4) grab the suggestion
    # best_lr = lr_finder.suggestion()
    # print(f"Suggested learning rate: {best_lr:.2e}")

    limit_batches = 1.0 if torch.cuda.is_available() else 0.1
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        fast_dev_run=False,
        logger=logger,
        max_epochs=config.num_epochs,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        accumulate_grad_batches=2,
        enable_progress_bar=not isinstance(logger, WandbLogger),
        precision="16-mixed",
        callbacks=[
            ckpt,
            LearningRateMonitor(logging_interval="epoch"),
            # swa,
        ],
        enable_model_summary=True,
        default_root_dir=config.model_folder,
    )

    trainer.fit(model, train_dataloader, valid_dataloader)

    # save_path = "/sc-scratch/sc-scratch-dh-face/physionet2025/loss-landscape-batch"

    # for i, batch in enumerate(train_dataloader):
    #     labels = batch["chagas"]
    #     labels = labels.view(-1)  # flattens to (batch_size,)
    #     if torch.sum(labels == 1).item() == labels.numel() // 2:
    #         print(f"Batch {i}: Exactly half the labels are 1s.")
    #         # Save the entire batch as a PyTorch .pt file
    #         torch.save(batch, os.path.join(save_path, 'batch.pt'))
    #         # Save the entire batch as a NumPy .npz file (convert all tensors to numpy)
    #         batch_np = {k: v.numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
    #         np.savez(os.path.join(save_path, 'batch.npz'), **batch_np)
    #         # Also save the model to the specified path
    #         save_model(save_path, model, config)
    #         break

    # saving one model correctly to only have weights
    # ckpt = LitResNet18NJ.load_from_checkpoint("/home/herzlerf/physionet-challenge-25/wandb/PhysioNetChallenge25/mhnli3m9/checkpoints/best-chagas-epoch=09-val/score=0.4363.ckpt", map_location="cpu", strict=False)
    # torch.save(ckpt, "/sc-scratch/sc-scratch-dh-face/physionet2025/evaluation_models/bce/pos1/model_weights_only.pt")

    # import shutil

    # file_path = "/home/herzlerf/physionet-challenge-25/wandb/PhysioNetChallenge25/vm85mq7m/checkpoints/best-chagas-epoch=28-val/score=0.4241.ckpt"
    # folder_path = "/sc-scratch/sc-scratch-dh-face/physionet2025/evaluation_models/ckpts/focal/sup_con"
    # model_path = os.path.join(folder_path, "model.ckpt")
    # # Create the folder if it does not exist
    # os.makedirs(folder_path, exist_ok=True)
    # shutil.copy(
    #     file_path,
    #     model_path
    # )

    if verbose:
        print("Done Training!")


def get_bn_loader(loader, device):
    for inputs, _ in loader:
        yield inputs.to(device)


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # Find all checkpoint files matching the naming pattern
    ckpt_files = glob.glob(os.path.join(model_folder, "best_ep_*_score_*.ckpt"))

    best_score = None
    best_ckpt = None

    score_pattern = re.compile(r"score_([0-9]+\.[0-9]{3})\.ckpt")

    for ckpt in ckpt_files:
        match = score_pattern.search(ckpt)
        if match:
            score = float(match.group(1))
            if best_score is None or score > best_score:
                best_score = score
                print(f"new best score: {best_score}")
                best_ckpt = ckpt

    if best_ckpt is not None:
        best_model_path = best_ckpt
    else:
        best_model_path = "tversky_43_m9au6yvp.ckpt"

    device = set_device()
    config = load_config(os.path.join(model_folder, "run_config.yaml"))
    # Check if there is a "model" key; if so, load from it, otherwise load the checkpoint directly
    # if "model" in checkpoint:
    #     model.load_state_dict(checkpoint["model"])
    # else:
    #     model.load_state_dict(checkpoint["state_dict"])
    model = LitResNet18NJ.load_from_checkpoint(
        best_model_path,
        map_location=device,
        strict=False,
        **{**asdict(config), "criterion": get_criterion(config)},
    )
    # Efficiently update model attributes from config if they differ
    # for key, value in asdict(config).items():
    #     if key == "criterion":
    #         setattr(model, key, get_criterion(config))
    #     elif hasattr(model, key):
    #         current_value = getattr(model, key)
    #         if current_value != value:
    #             setattr(model, key, value)

    # print("Loaded model parameters from model object:")
    # for k in asdict(config).keys():
    #     if k == "criterion":
    #         print(f"  {k}: {get_criterion(config)}")
    #     elif hasattr(model, k):
    #         print(f"  {k}: {getattr(model, k)}")

    tqdm.write(f"Model loaded from {best_model_path} onto {device}")

    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    try:
        W = 2500  # keep equal to training crop size
        STRIDE = W // 4  # sliding stride
        N_MAX = 7  # cap number of crops

        def pad_center(x, target_len):
            L = x.shape[-1]
            if L >= target_len:
                return x
            pad = target_len - L
            left = pad // 2
            right = pad - left
            return F.pad(x, (left, right))

        def crop_starts(L, W, stride=STRIDE, n_max=N_MAX):
            if L <= W:
                return [0]
            starts = list(range(0, L - W + 1, max(1, stride)))
            if starts[-1] != L - W:
                starts.append(L - W)
            if len(starts) > n_max:
                # downsample to n_max evenly spaced starts
                idxs = [
                    round(i * (len(starts) - 1) / (n_max - 1)) for i in range(n_max)
                ]
                starts = [starts[i] for i in idxs]
            return starts

        def forward_logits(m, x_crop):
            out = m(x_crop)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            return out.squeeze(0)  # (num_classes,)

        device = set_device()
        model.to(device)
        model.eval()

        signal, _ = preprocess_ecg_safe(record, target_sample_rate=400)  # (C, L) numpy

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
        x = (
            torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        )  # (1, C, L)
        L = x.shape[-1]
        if L <= W:
            xw = pad_center(x, W)
            logits = forward_logits(model, xw)
            # Print spread for single crop (just one value)
            # print(f"Prediction spread (single crop): {logits.detach().cpu().numpy()}")
        else:
            starts = crop_starts(L, W)
            crop_logits = []
            for s in starts:
                crop = x[..., s : s + W].contiguous()
                crop_logits.append(forward_logits(model, crop))
            crop_logits_tensor = torch.stack(
                crop_logits, 0
            )  # shape: (num_crops, num_classes)
            logits = crop_logits_tensor.max(dim=0).values  # per-class max over crops
            # Print spread for all crops
            # crop_preds = (
            #     torch.sigmoid(crop_logits_tensor).detach().cpu().numpy().flatten()
            # )
            # print(f"Prediction spread (all crops): min={crop_preds.min():.4f}, max={crop_preds.max():.4f}, mean={crop_preds.mean():.4f}, std={crop_preds.std():.4f}, values={crop_preds}")

        prob = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        probability_output = float(prob[0])  # binary case
        binary_output = int(probability_output > 0.5)

        return binary_output, probability_output

    except Exception as e:
        if verbose:
            print(f"Error processing {record}: {e}")
        return 0, 0.0


def run_model_old(record, model, verbose):
    """
    Loads a single ECG file, preprocesses it, and runs inference.

    Args:
    - model (torch.nn.Module): The trained PyTorch model.
    - file_path (str): Path to the ECG `.dat` file.
    - device (torch.device): The device (CPU/GPU) to run inference on.

    Returns:
    - torch.Tensor: Model output (prediction).
    """
    try:
        device = set_device()
        model.to(device)
        model.eval()
        # Load the ECG signal
        # signal, _ = wfdb.rdsamp(record)
        # # Preprocess the signal (Ensure correct shape)
        # signal = np.nan_to_num(signal).T  # Convert NaNs to 0 and transpose to (channels, timesteps)
        # signal = ensure_12_channels(signal, desired_channels=12)  # Ensure 12 channels
        signal, _ = preprocess_ecg_safe(record, target_sample_rate=400)
        # from stats import MEAN, STD  # load mean and std from stats.py
        # signal = (signal - MEAN[:, None]) / STD[:, None]  # normalize signal
        # Center crop or pad the signal to 2048 samples
        signal = RandomCropOrPad(target_length=3072)(
            torch.tensor(signal, dtype=torch.float32)
        )
        # Convert to PyTorch tensor and move to device
        signal_tensor = signal.unsqueeze(0).to(device)
        # Run inference
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = model(signal_tensor)  # Forward pass
            output = output[0]  # get logits
            probability_output = torch.sigmoid(output).cpu().numpy().flatten()[0]
            binary_output = int(probability_output > 0.5)
        return binary_output, probability_output
    except Exception as e:
        if verbose:
            print(f"Error processing {record}: {e}")
        return 0, 0.0


def ensure_12_channels(signal: np.ndarray, desired_channels: int = 12) -> np.ndarray:
    """Ensure signal has exactly 12 channels by padding or truncating."""
    current_channels = signal.shape[0]
    if current_channels < desired_channels:
        pad_width = desired_channels - current_channels
        signal = np.pad(signal, ((0, pad_width), (0, 0)), mode="constant")
    elif current_channels > desired_channels:
        signal = signal[:desired_channels, :]
    return signal


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def set_seeds(seed=42):
    """
    Set the random seed for reproducibility across NumPy and PyTorch.

    This function sets the seed for NumPy's random number generator, PyTorch's
    CPU and GPU random number generators, and ensures deterministic behavior
    for CUDA operations.

    Args:
        seed (int, optional): The seed value to set. Defaults to 42.

    Returns:
        int: The seed value that was set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seeds set to {seed}")
    return seed


def set_device(use_cuda=None, verbose=False):
    """
    Sets the device to be used for PyTorch operations.

    Parameters:
    use_cuda (bool, optional): If True, forces the use of CUDA (GPU).
                               If False, forces the use of CPU.
                               If None (default), automatically detects
                               if CUDA is available and uses it if possible.
    verbose (bool, optional): If True, prints the selected device. Default is False.

    Returns:
    torch.device: The device object representing the selected device.
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print(f"Using device: {device}")
    return device


def create_model(
    config: Config,
    input_dim=(12, 4096),
    seq_length=[4096, 1024, 256, 64, 16],
    filter_size=[64, 128, 196, 256, 320],
    n_classes=1,
    kernel_size=17,
    dropout_rate=0.5,
):
    """
    Create a 1D ResNet model for sequence classification.

    Args:
        input_dim (tuple, optional): The dimensions of the input data, where the first value
            represents the number of channels and the second value represents the sequence length.
            Defaults to (12, 4096).
        seq_length (list, optional): A list of sequence lengths for each ResNet block.
            Defaults to [4096, 1024, 256, 64, 16].
        filter_size (list, optional): A list of filter sizes for each ResNet block.
            Defaults to [64, 128, 196, 256, 320].
        n_classes (int, optional): The number of output classes for classification.
            Defaults to 1.
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 17.
        dropout_rate (float, optional): The dropout rate to use in the model.
            Defaults to 0.5.

    Returns:
        ResNet1d: An instance of the ResNet1d model configured with the specified parameters.
    """
    criterion = get_criterion(config)

    tqdm.write(f"am i using supcon? {config.use_sup_con}")
    tqdm.write(f"am i using prototypes? {config.use_prototypes}")

    model = LitResNet18NJ(
        num_classes=1,
        channels=12,
        lr=config.lr,
        lr_scheduler=config.lr_scheduler,
        optimizer=config.optimizer,
        momentum=config.momentum,
        classifier_weight_decay=config.classifier_weight_decay,
        params_weight_decay=config.params_weight_decay,
        block=BasicBlock,
        norm_type=config.norm_type,
        norm_groups=config.norm_groups,
        layers=[2, 2, 2, 2],
        criterion=criterion,
        crop_size=config.crop_size,
        max_time_warp=config.max_time_warp,
        classifier_weight=config.classifier_weight,
        use_sup_con=config.use_sup_con,
        use_prototypes=config.use_prototypes,
        sup_con_weight=config.sup_con_weight,
        sup_con_temp=config.sup_con_temp,
        dropout_rate=config.dropout_rate,
        se_reduction=None if config.se_reduction == 0 else config.se_reduction,
    )

    print("Created new model!")

    return model


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


# Save your trained model.
def save_model(model_folder, model, config=None):
    """
    Save a PyTorch model and its configuration to the specified folder.

    Args:
        model_folder (str): The path to the folder where the model and configuration will be saved.
        model (torch.nn.Module): The PyTorch model to be saved.
        config (dict): A dictionary containing the configuration settings to be saved.

    Saves:
        - The model's state dictionary as 'best_model.ckpt' in the specified folder.
        - The configuration dictionary as 'config.json' in the specified folder.
    """
    # save the PyTorch model and config file
    torch.save(model.state_dict(), os.path.join(model_folder, "best_model.ckpt"))
    # save config as yaml
    if config:
        with open(os.path.join(model_folder, "config.yaml"), "w") as f:
            yaml.dump(config, f)
