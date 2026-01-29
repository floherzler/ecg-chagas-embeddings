from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Subset

from ecg_chagas_embeddings.data.dataset import collate_dict_batch


@dataclass(frozen=True)
class DFTLRPConfig:
    leverage_symmetry: bool = True
    precision: int = 32
    epsilon: float = 1e-6
    freq_max_hz: float = 45.0

    # Only used for optional time-frequency visualizations (can be memory-heavy).
    compute_timefreq: bool = False
    window_width: int = 128
    window_shift: int = 128
    window_shape: str = "rectangle"  # rectangle|halfsine


def _as_dftlrp_config(cfg: Optional[Mapping[str, Any]]) -> DFTLRPConfig:
    if cfg is None:
        return DFTLRPConfig()
    return DFTLRPConfig(
        leverage_symmetry=bool(cfg.get("leverage_symmetry", True)),
        precision=int(cfg.get("precision", 32)),
        epsilon=float(cfg.get("epsilon", 1e-6)),
        freq_max_hz=float(cfg.get("freq_max_hz", 45.0)),
        compute_timefreq=bool(cfg.get("compute_timefreq", False)),
        window_width=int(cfg.get("window_width", 128)),
        window_shift=int(cfg.get("window_shift", 1)),
        window_shape=str(cfg.get("window_shape", "rectangle")),
    )


def _parse_bands_hz(
    bands_hz: Optional[Sequence[Sequence[float]]],
) -> List[Tuple[float, float]]:
    if not bands_hz:
        return [(0.67, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 45.0)]
    out: List[Tuple[float, float]] = []
    for band in bands_hz:
        if len(band) != 2:
            raise ValueError(f"bands_hz entries must be [low, high], got {band}")
        low, high = float(band[0]), float(band[1])
        if not (high > low):
            raise ValueError(f"Invalid band range: {band}")
        out.append((low, high))
    return out


def extract_pos_logit(model_out: Any) -> torch.Tensor:
    """
    Return a 1D tensor of positive-class logits (shape [B]).

    Supports:
      - Tensor logits [B], [B,1], [B,2]
      - Tuple where last element is logits (LitResNet18: (feats, proj, logits))
      - Dict containing a 'logits' key
    """
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        model_out = model_out[-1]
    if isinstance(model_out, dict):
        if "logits" not in model_out:
            raise ValueError(
                f"Model output dict missing 'logits' key: {list(model_out)}"
            )
        model_out = model_out["logits"]
    if not torch.is_tensor(model_out):
        raise TypeError(f"Unsupported model output type: {type(model_out)}")

    logits = model_out
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, 1]
    raise ValueError(f"Unsupported logits shape for pos_logit: {tuple(logits.shape)}")


def _repo_root_from_here() -> Path:
    # src/ecg_chagas_embeddings/callbacks/xai_probe.py -> repo root
    return Path(__file__).resolve().parents[3]


def _import_dft_lrp() -> Any:
    """
    Import the upstream DFT-LRP code from the `external/dft-lrp` submodule.
    """
    repo_root = _repo_root_from_here()
    code_root = repo_root / "external" / "dft-lrp" / "code"
    if not code_root.exists():
        raise FileNotFoundError(
            f"Missing DFT-LRP submodule at {code_root}. "
            "Initialize it with `git submodule update --init --recursive`."
        )
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    import dft_lrp  # type: ignore

    return dft_lrp


def compute_lrp_relevance_time(
    *,
    pl_module: torch.nn.Module,
    x: torch.Tensor,
    zennit_composite: str = "EpsilonPlus",
    rel_is_model_out: bool = True,
) -> torch.Tensor:
    """
    Compute time-domain relevance via zennit LRP for the positive logit.

    Returns relevance tensor with same shape as input x: [B,C,T].
    """
    try:
        import zennit.composites  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError(
            "zennit is required for DFT-LRP (time-domain LRP). "
            "Install with `uv add zennit`."
        ) from exc

    if zennit_composite == "EpsilonPlus":
        composite = zennit.composites.EpsilonPlus()
    elif zennit_composite == "EpsilonAlpha2Beta1":
        composite = zennit.composites.EpsilonAlpha2Beta1()
    else:
        raise ValueError(f"Unsupported zennit composite: {zennit_composite}")

    x_leaf = x.detach().requires_grad_(True)
    composite.register(pl_module)
    try:
        out = extract_pos_logit(pl_module(x_leaf))
        target_out = out.detach() if rel_is_model_out else torch.sign(out.detach())
        relevance = torch.autograd.grad(
            out, x_leaf, grad_outputs=target_out, retain_graph=False
        )[0]
    finally:
        composite.remove()
    return relevance


def compute_dft_band_fractions_from_relevance(
    *,
    relevance_freq: torch.Tensor,
    fs_hz: float,
    signal_length: int,
    freq_max_hz: float,
    bands_hz: Sequence[Tuple[float, float]],
    per_lead: bool,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Compute relevance mass fractions per frequency band from DFT-LRP relevance in frequency domain.

    Args:
      relevance_freq: [B,L,F] where F = signal_length//2 + 1 (rfft bins)
    """
    if relevance_freq.ndim != 3:
        raise ValueError(
            f"Expected relevance_freq [B,L,F], got {tuple(relevance_freq.shape)}"
        )
    B, L, F = relevance_freq.shape
    expected_f = signal_length // 2 + 1
    if F != expected_f:
        raise ValueError(
            f"Expected F={expected_f} for signal_length={signal_length}, got F={F}"
        )

    freqs = torch.fft.rfftfreq(signal_length, d=1.0 / float(fs_hz)).to(
        device=relevance_freq.device
    )
    max_hz = float(freq_max_hz)
    min_low = min(float(lo) for lo, _hi in bands_hz)
    mask_total = (freqs >= min_low) & (freqs < max_hz)

    mass = relevance_freq.abs()
    total_mass = mass[:, :, mask_total].sum(dim=(1, 2))  # [B]
    denom = total_mass.clamp_min(eps)

    band_masses_pooled: List[torch.Tensor] = []
    band_masses_per_lead: List[torch.Tensor] = []
    for lo, hi in bands_hz:
        mask = (freqs >= float(lo)) & (freqs < float(hi)) & (freqs < max_hz)
        mass_per_lead = mass[:, :, mask].sum(dim=2)  # [B,L]
        band_masses_per_lead.append(mass_per_lead)
        band_masses_pooled.append(mass_per_lead.sum(dim=1))  # [B]

    pooled_fracs = torch.stack(band_masses_pooled, dim=1) / denom.unsqueeze(1)  # [B,nb]
    per_lead_fracs_out: Optional[torch.Tensor] = None
    if per_lead:
        per_lead_masses = torch.stack(band_masses_per_lead, dim=2)  # [B,L,nb]
        lead_denom = mass[:, :, mask_total].sum(dim=2).clamp_min(eps)  # [B,L]
        per_lead_fracs_out = per_lead_masses / lead_denom.unsqueeze(2)

    return pooled_fracs, per_lead_fracs_out, total_mass


def lead_entropy_from_relevance_time(
    relevance_time: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Normalized entropy of per-lead relevance mass, in [0,1], shape [B].
    Lower => model focuses on fewer leads.
    """
    if relevance_time.ndim != 3:
        raise ValueError(
            f"Expected relevance_time [B,L,T], got {tuple(relevance_time.shape)}"
        )
    mass = relevance_time.abs().sum(dim=2)  # [B,L]
    p = mass / mass.sum(dim=1, keepdim=True).clamp_min(eps)
    h = -(p * (p.clamp_min(eps).log())).sum(dim=1)
    return h / math.log(relevance_time.shape[1])


def _read_ids_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError(f"Expected JSON list in {path}")
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts.extend([p for p in line.replace(",", " ").split() if p])
    return [str(p) for p in parts]


def _default_run_dir(trainer) -> Path:
    logger = getattr(trainer, "logger", None)
    if logger is not None and hasattr(logger, "experiment"):
        exp = logger.experiment
        run_dir = getattr(exp, "dir", None)
        if run_dir:
            return Path(run_dir)
    log_dir = getattr(trainer, "log_dir", None)
    if log_dir:
        return Path(log_dir)
    root_dir = getattr(trainer, "default_root_dir", None)
    if root_dir:
        return Path(root_dir)
    return Path.cwd()


class XAIProbeCallback(Callback):
    """
    DFT-LRP probe: compute time-domain LRP relevances via zennit and propagate to frequency / time-frequency via DFT-LRP.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        every_n_epochs: int = 5,
        n_pos: int = 64,
        n_neg: int = 64,
        seed: int = 1337,
        fs_hz: float = 400.0,
        target: str = "pos_logit",
        dft_lrp: Optional[Mapping[str, Any]] = None,
        zennit_composite: str = "EpsilonPlus",
        rel_is_model_out: bool = True,
        bands_hz: Optional[Sequence[Sequence[float]]] = None,
        per_lead: bool = True,
        log_heatmaps: bool = False,
        num_example_plots: int = 8,
        probe_ids_path: Optional[str] = None,
        probe_batch_size: int = 16,
        eps: float = 1e-12,
        save_per_sample_csv: bool = True,
        log_wandb_artifact: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.every_n_epochs = int(max(1, every_n_epochs))
        self.n_pos = int(max(0, n_pos))
        self.n_neg = int(max(0, n_neg))
        self.seed = int(seed)
        self.fs_hz = float(fs_hz)
        self.target = str(target)
        self.dft_lrp_cfg = _as_dftlrp_config(dft_lrp)
        self.zennit_composite = str(zennit_composite)
        self.rel_is_model_out = bool(rel_is_model_out)
        self.bands_hz = _parse_bands_hz(bands_hz)
        self.per_lead = bool(per_lead)
        self.log_heatmaps = bool(log_heatmaps)
        self.num_example_plots = int(max(0, num_example_plots))
        self.probe_ids_path = str(probe_ids_path) if probe_ids_path else None
        self.probe_batch_size = int(max(1, probe_batch_size))
        self.eps = float(eps)
        self.save_per_sample_csv = bool(save_per_sample_csv)
        self.log_wandb_artifact = bool(log_wandb_artifact)

        self._probe_ids: Optional[List[str]] = None
        self._probe_loader: Optional[DataLoader] = None
        self._example_ids: Optional[List[str]] = None

        self._dftlrp_dft = None
        self._dftlrp_stdft = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if not self.enabled:
            return
        if stage not in ("fit", "validate"):
            return
        self._ensure_probe_loader(trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        epoch = int(getattr(trainer, "current_epoch", 0))
        should_run = (epoch == 0) or ((epoch + 1) % self.every_n_epochs == 0)
        if not should_run:
            return
        if self.target != "pos_logit":
            raise ValueError(
                f"Unsupported target: {self.target} (only 'pos_logit' supported)"
            )

        self._ensure_probe_loader(trainer)
        if self._probe_loader is None:
            return

        metrics, df_epoch = self._run_probe(trainer, pl_module, self._probe_loader)
        if metrics:
            self._log_metrics(trainer, metrics)
        self._maybe_save_outputs(trainer, epoch, df_epoch)
        if self.log_heatmaps:
            self._maybe_log_heatmaps(trainer, pl_module, df_epoch)

    def _ensure_probe_loader(self, trainer) -> None:
        if self._probe_loader is not None:
            return

        val_loader = None
        if getattr(trainer, "datamodule", None) is not None:
            try:
                val_loader = trainer.datamodule.val_dataloader()
            except Exception:
                val_loader = None
        if val_loader is None:
            v = getattr(trainer, "val_dataloaders", None)
            if v:
                val_loader = v[0] if isinstance(v, (list, tuple)) else v
        if val_loader is None:
            return

        dataset = getattr(val_loader, "dataset", None)
        if dataset is None:
            return

        ids = self._select_probe_ids(dataset)
        if not ids:
            return

        id_to_index: Dict[str, int] = {}
        if hasattr(dataset, "metadata"):
            meta = dataset.metadata
            if hasattr(meta, "__getitem__") and "exam_id" in meta:
                for i, v in enumerate(meta["exam_id"].tolist()):
                    sid = str(v)
                    if sid not in id_to_index:
                        id_to_index[sid] = int(i)
        if not id_to_index:
            for i in range(len(dataset)):
                sample = dataset[i]
                sid = (
                    str(sample.get("exam_id", i))
                    if isinstance(sample, dict)
                    else str(i)
                )
                if sid not in id_to_index:
                    id_to_index[sid] = i

        indices = [id_to_index[sid] for sid in ids if sid in id_to_index]
        subset = Subset(dataset, indices)
        self._probe_loader = DataLoader(
            subset,
            batch_size=self.probe_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_dict_batch,
            drop_last=False,
        )
        self._probe_ids = ids
        self._example_ids = ids[: self.num_example_plots]

        run_dir = _default_run_dir(trainer) / "artifacts" / "xai_probe"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "xai_probe_ids.json").write_text(
            json.dumps(ids, indent=2), encoding="utf-8"
        )

    def _select_probe_ids(self, dataset) -> List[str]:
        ids_from_file: Optional[List[str]] = None
        if self.probe_ids_path:
            p = Path(self.probe_ids_path)
            if p.exists():
                ids_from_file = _read_ids_file(p)
            else:
                raise FileNotFoundError(f"probe_ids_path does not exist: {p}")

        meta = getattr(dataset, "metadata", None)
        if meta is None or "exam_id" not in meta or "chagas" not in meta:
            if ids_from_file is None:
                raise ValueError(
                    "XAIProbeCallback requires a validation dataset with a pandas-like "
                    "`metadata` containing columns ['exam_id','chagas'], or a `probe_ids_path`."
                )
            return ids_from_file

        df = meta[["exam_id", "chagas"]].copy()
        df["exam_id"] = df["exam_id"].map(lambda x: str(x))
        df["label"] = (df["chagas"].astype(float) > 0.5).astype(int)

        if ids_from_file is not None:
            return [str(x) for x in ids_from_file]

        rng = np.random.default_rng(self.seed)
        pos = df[df["label"] == 1]["exam_id"].to_numpy(dtype=object)
        neg = df[df["label"] == 0]["exam_id"].to_numpy(dtype=object)

        n_pos = min(self.n_pos, int(pos.size))
        n_neg = min(self.n_neg, int(neg.size))
        if n_pos == 0 and n_neg == 0:
            return []

        chosen_pos = rng.permutation(np.sort(pos.astype(str)))[:n_pos].tolist()
        chosen_neg = rng.permutation(np.sort(neg.astype(str)))[:n_neg].tolist()
        return [str(x) for x in (chosen_pos + chosen_neg)]

    def _get_dftlrp(
        self, *, signal_length: int, device: torch.device, short_time: bool
    ):
        dft_lrp = _import_dft_lrp()
        cuda = device.type == "cuda"
        if short_time:
            if self._dftlrp_stdft is None:
                self._dftlrp_stdft = dft_lrp.DFTLRP(
                    signal_length,
                    leverage_symmetry=self.dft_lrp_cfg.leverage_symmetry,
                    precision=self.dft_lrp_cfg.precision,
                    cuda=cuda,
                    window_shift=self.dft_lrp_cfg.window_shift,
                    window_width=self.dft_lrp_cfg.window_width,
                    window_shape=self.dft_lrp_cfg.window_shape,
                    create_dft=False,
                    create_inverse=False,
                )
            return self._dftlrp_stdft

        if self._dftlrp_dft is None:
            self._dftlrp_dft = dft_lrp.DFTLRP(
                signal_length,
                leverage_symmetry=self.dft_lrp_cfg.leverage_symmetry,
                precision=self.dft_lrp_cfg.precision,
                cuda=cuda,
                create_stdft=False,
                create_inverse=False,
            )
        return self._dftlrp_dft

    def _run_probe(
        self, trainer, pl_module, loader: DataLoader
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        device = getattr(pl_module, "device", torch.device("cpu"))
        pl_module.eval()

        rows: List[Dict[str, Any]] = []
        pooled_fracs_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []
        lead_ent_all: List[torch.Tensor] = []
        near_cutoff_all: List[torch.Tensor] = []
        total_mass_all: List[torch.Tensor] = []

        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.autocast(device_type=str(device).split(":")[0], enabled=False),
        ):
            for batch in loader:
                y = batch["chagas"].view(-1).to(device=device, dtype=torch.float32)
                ids = batch.get("exam_id", None)
                x = batch.get("ecg")
                if x is None and "ecg_views" in batch:
                    x = batch["ecg_views"][:, 0]
                if x is None:
                    raise ValueError("Probe batch requires 'ecg' or 'ecg_views'.")
                x = x.to(device=device, dtype=torch.float32)

                relevance_time = compute_lrp_relevance_time(
                    pl_module=pl_module,
                    x=x,
                    zennit_composite=self.zennit_composite,
                    rel_is_model_out=self.rel_is_model_out,
                )

                B, L, T = relevance_time.shape
                dftlrp = self._get_dftlrp(
                    signal_length=T, device=device, short_time=False
                )

                x_np = x.detach().cpu().numpy().reshape(B * L, T)
                rel_np = relevance_time.detach().cpu().numpy().reshape(B * L, T)
                _signal_freq, relevance_freq = dftlrp.dft_lrp(
                    rel_np,
                    x_np,
                    real=False,
                    short_time=False,
                    epsilon=float(self.dft_lrp_cfg.epsilon),
                )
                relevance_freq_t = (
                    torch.from_numpy(np.asarray(relevance_freq))
                    .to(device=device, dtype=torch.float32)
                    .reshape(B, L, -1)
                )

                pooled_fracs, _per_lead_fracs, total_mass = (
                    compute_dft_band_fractions_from_relevance(
                        relevance_freq=relevance_freq_t,
                        fs_hz=self.fs_hz,
                        signal_length=T,
                        freq_max_hz=self.dft_lrp_cfg.freq_max_hz,
                        bands_hz=self.bands_hz,
                        per_lead=self.per_lead,
                        eps=self.eps,
                    )
                )
                lead_ent = lead_entropy_from_relevance_time(
                    relevance_time, eps=self.eps
                )

                def _band_idx(lo: float, hi: float) -> int:
                    for i, (a, b) in enumerate(self.bands_hz):
                        if abs(a - lo) < 1e-9 and abs(b - hi) < 1e-9:
                            return i
                    return -1

                i_low = _band_idx(0.67, 2.0)
                i_mid = _band_idx(5.0, 15.0)
                if i_low != -1 and i_mid != -1:
                    near_cutoff = pooled_fracs[:, i_low] / (
                        pooled_fracs[:, i_mid] + self.eps
                    )
                else:
                    near_cutoff = torch.full(
                        (pooled_fracs.shape[0],), float("nan"), device=device
                    )

                pooled_fracs_all.append(pooled_fracs.detach().cpu())
                labels_all.append((y > 0.5).to(torch.int64).detach().cpu())
                lead_ent_all.append(lead_ent.detach().cpu())
                near_cutoff_all.append(near_cutoff.detach().cpu())
                total_mass_all.append(total_mass.to(torch.float32).detach().cpu())

                ids_list: List[str]
                if ids is None:
                    ids_list = [
                        f"idx_{len(rows) + i}" for i in range(pooled_fracs.shape[0])
                    ]
                elif isinstance(ids, (list, tuple)):
                    ids_list = [str(x) for x in ids]
                elif isinstance(ids, torch.Tensor):
                    ids_list = [str(x) for x in ids.detach().cpu().tolist()]
                else:
                    ids_list = [str(ids)]

                for i in range(pooled_fracs.shape[0]):
                    row: Dict[str, Any] = {
                        "sample_id": ids_list[i],
                        "label": int((y[i] > 0.5).item()),
                        "lead_entropy": float(lead_ent[i].detach().cpu().item()),
                        "near_cutoff_ratio": float(
                            near_cutoff[i].detach().cpu().item()
                        ),
                        "total_mass": float(total_mass[i].detach().cpu().item()),
                    }
                    for b, (lo, hi) in enumerate(self.bands_hz):
                        row[f"rel_frac_{lo}_{hi}"] = float(
                            pooled_fracs[i, b].detach().cpu().item()
                        )
                    rows.append(row)

        df_epoch = pd.DataFrame(rows)
        if pooled_fracs_all:
            pooled_all = torch.cat(pooled_fracs_all, dim=0)  # [N,nb]
            y_all = torch.cat(labels_all, dim=0)  # [N]
            lead_ent = torch.cat(lead_ent_all, dim=0)
            near_cutoff = torch.cat(near_cutoff_all, dim=0)
            total_mass = torch.cat(total_mass_all, dim=0)
        else:
            pooled_all = torch.zeros((0, len(self.bands_hz)))
            y_all = torch.zeros((0,), dtype=torch.int64)
            lead_ent = torch.zeros((0,))
            near_cutoff = torch.zeros((0,))
            total_mass = torch.zeros((0,))

        metrics: Dict[str, float] = {}
        epoch = int(getattr(trainer, "current_epoch", 0))
        metrics["xai/epoch"] = float(epoch)
        metrics["xai/n_samples"] = float(pooled_all.shape[0])

        def add_band_metrics(prefix: str, fracs: torch.Tensor) -> None:
            if fracs.numel() == 0:
                return
            for b, (lo, hi) in enumerate(self.bands_hz):
                metrics[f"{prefix}/rel_frac_{lo}_{hi}"] = float(
                    fracs[:, b].mean().item()
                )

        add_band_metrics("xai", pooled_all)
        if pooled_all.numel() > 0:
            metrics["xai/lead_entropy"] = float(lead_ent.mean().item())
            metrics["xai/near_cutoff_ratio"] = float(torch.nanmean(near_cutoff).item())
            metrics["xai/total_mass"] = float(total_mass.mean().item())

        if pooled_all.numel() > 0:
            pos_mask = y_all == 1
            neg_mask = y_all == 0
            if pos_mask.any():
                add_band_metrics("xai_pos", pooled_all[pos_mask])
            if neg_mask.any():
                add_band_metrics("xai_neg", pooled_all[neg_mask])

        return metrics, df_epoch

    def _log_metrics(self, trainer, metrics: Dict[str, float]) -> None:
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return
        step = int(getattr(trainer, "global_step", 0))
        try:
            logger.log_metrics(metrics, step=step)
            logger.save()
        except Exception:
            pass

    def _maybe_save_outputs(self, trainer, epoch: int, df_epoch: pd.DataFrame) -> None:
        run_dir = _default_run_dir(trainer) / "artifacts" / "xai_probe"
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.save_per_sample_csv and not df_epoch.empty:
            out_path = run_dir / f"xai_probe_epoch_{epoch:04d}.csv"
            df_epoch.to_csv(out_path, index=False)
            if self.log_wandb_artifact:
                self._log_csv_as_wandb_artifact(trainer, out_path, epoch)

    def _log_csv_as_wandb_artifact(self, trainer, csv_path: Path, epoch: int) -> None:
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        exp = logger.experiment
        if exp is None:
            return
        try:
            import wandb  # type: ignore
        except Exception:
            return
        try:
            art = wandb.Artifact(name=f"xai_probe_epoch_{epoch:04d}", type="xai_probe")
            art.add_file(str(csv_path))
            exp.log_artifact(art)
        except Exception:
            pass

    def _maybe_log_heatmaps(self, trainer, pl_module, df_epoch: pd.DataFrame) -> None:
        if not self.dft_lrp_cfg.compute_timefreq:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        if self._example_ids is None or len(self._example_ids) == 0:
            return
        if self._probe_loader is None:
            return
        try:
            import wandb  # type: ignore
        except Exception:
            return
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
            from matplotlib.figure import Figure  # type: ignore
        except Exception:
            return

        device = getattr(pl_module, "device", torch.device("cpu"))
        pl_module.eval()

        want = set(self._example_ids)
        examples: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for batch in self._probe_loader:
            ids = batch.get("exam_id", None)
            if ids is None:
                continue
            if isinstance(ids, torch.Tensor):
                ids_list = [str(x) for x in ids.detach().cpu().tolist()]
            elif isinstance(ids, (list, tuple)):
                ids_list = [str(x) for x in ids]
            else:
                ids_list = [str(ids)]
            if not any(i in want for i in ids_list):
                continue
            x = batch.get("ecg")
            if x is None and "ecg_views" in batch:
                x = batch["ecg_views"][:, 0]
            if x is None:
                continue
            x = x.to(device=device, dtype=torch.float32)

            with (
                torch.inference_mode(False),
                torch.enable_grad(),
                torch.autocast(device_type=str(device).split(":")[0], enabled=False),
            ):
                relevance_time = compute_lrp_relevance_time(
                    pl_module=pl_module,
                    x=x,
                    zennit_composite=self.zennit_composite,
                    rel_is_model_out=self.rel_is_model_out,
                )

            for i, sid in enumerate(ids_list):
                if sid in want:
                    examples.append(
                        (sid, x[i].detach().cpu(), relevance_time[i].detach().cpu())
                    )
            if len(examples) >= len(want):
                break

        if not examples:
            return

        images = {}
        dftlrp = self._get_dftlrp(
            signal_length=examples[0][1].shape[-1], device=device, short_time=True
        )
        fs = float(self.fs_hz)
        T = int(examples[0][1].shape[-1])
        freqs = np.fft.rfftfreq(T, d=1.0 / fs)
        fmask = freqs <= float(self.dft_lrp_cfg.freq_max_hz)
        k_max = (
            int(np.where(fmask)[0][-1]) if np.any(fmask) else min(100, len(freqs) - 1)
        )

        for sid, x_one, rel_one in examples[: self.num_example_plots]:
            # Lead-sum for readability.
            x_sum = x_one.sum(dim=0, keepdim=True).numpy()  # [1,T]
            r_sum = rel_one.sum(dim=0, keepdim=True).numpy()  # [1,T]
            _sig_tf, rel_tf = dftlrp.dft_lrp(
                r_sum,
                x_sum,
                real=False,
                short_time=True,
                epsilon=float(self.dft_lrp_cfg.epsilon),
            )
            rel_tf = np.asarray(rel_tf)[0, :, :k_max]  # [W,K]

            fig = Figure(figsize=(7.0, 3.5), dpi=180)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(rel_tf.T, aspect="auto", origin="lower")
            ax.set_title(f"DFT-LRP relevance (STDFT, lead-sum)\n{sid}")
            ax.set_xlabel("window index")
            ax.set_ylabel("k (freq bins)")
            fig.tight_layout()
            canvas.draw()
            img = np.asarray(canvas.buffer_rgba())[:, :, :3]
            images[f"xai/heatmap_{sid}"] = wandb.Image(img)

        if images:
            try:
                logger.experiment.log(
                    images, step=int(getattr(trainer, "global_step", 0))
                )
            except Exception:
                pass
