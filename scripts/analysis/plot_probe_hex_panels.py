#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.nanmedian(x))


def _nanmedian_any(x: np.ndarray | list[float]) -> float:
    """Accept either ndarray or list[float] (matches matplotlib typing) and delegate to _nanmedian."""
    return _nanmedian(np.asarray(x, dtype=float))


def _prep_binary(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float).to_numpy(dtype=float)
    s_num = pd.to_numeric(series, errors="coerce")
    out = np.full(len(series), np.nan, dtype=float)
    if s_num.notna().any():
        v = s_num.to_numpy(dtype=float)
        out[v == 0] = 0.0
        out[v == 1] = 1.0
        return out
    s_str = series.astype("string[python]").str.strip().str.lower()
    out[s_str.isin(["1", "true", "t", "yes", "y"])] = 1.0
    out[s_str.isin(["0", "false", "f", "no", "n"])] = 0.0
    return out


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    pad = 0.02
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    dx = xmax - xmin
    dy = ymax - ymin
    return (xmin - pad * dx, xmax + pad * dx, ymin - pad * dy, ymax + pad * dy)


def _square_extent(
    ext: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Square extent centered at the original extent center.

    This intentionally adds empty space on the shorter axis so each subplot can be square
    and still use equal x/y scaling (avoids "squished" UMAP panels and keeps hexagons regular).
    """
    xmin, xmax, ymin, ymax = ext
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    r = max(float(xmax - xmin), float(ymax - ymin))
    # Add a small extra margin so hexes don't touch the axes frame.
    r *= 1.04
    half = 0.5 * r
    return (xmid - half, xmid + half, ymid - half, ymid + half)


def _format_run_title(*, run_id: str, space: str, method: str) -> str:
    """
    Produce a thesis-friendly title from a run_id like:
      t1-exp01-bp-bce-rot10
    """
    tokens = [t for t in str(run_id).split("-") if t]
    track = None
    exp = None
    preprocessing = None
    loss = None
    rot_deg = None
    other: list[str] = []

    for t in tokens:
        if t in {"t1", "t2", "t3"}:
            track = t.lower()
            continue
        if t.startswith("exp") and len(t) > 3:
            exp = t.lower()
            continue
        if t.startswith("rot"):
            try:
                rot_deg = int(t[len("rot") :])
                continue
            except Exception:
                other.append(t)
                continue
        if preprocessing is None and t in {"bp", "softclip", "zscore"}:
            preprocessing = t
            continue
        if loss is None and t in {"bce", "tversky", "ce"}:
            loss = t
            continue
        other.append(t)

    pre_map = {"bp": "BP", "softclip": "Softclip", "zscore": "Z-score"}
    loss_map = {"bce": "BCE", "tversky": "Tversky", "ce": "CE"}
    space_map = {"enc": "Enc", "proj": "Proj"}
    method_map = {"umap": "UMAP", "pca": "PCA"}

    parts: list[str] = []
    if track and exp:
        # Compact: Exp 1-01 (track 1, experiment 01)
        try:
            tnum = int(track.lstrip("t"))
            enum = int(exp.lstrip("exp"))
            parts.append(f"Exp {tnum}-{enum:02d}")
        except Exception:
            parts.append(f"{track}-{exp}")
    else:
        if track:
            parts.append(track.upper())
        if exp:
            parts.append(exp.upper())
    if preprocessing:
        parts.append(pre_map.get(preprocessing, preprocessing))
    if loss:
        parts.append(loss_map.get(loss, loss.upper()))
    if rot_deg is not None:
        parts.append(f"Rot {rot_deg}°")
    if other:
        parts.append(" ".join(other))

    emb = f"{space_map.get(space, space)} {method_map.get(method, method.upper())}"
    if parts:
        return " · ".join(parts) + " — " + emb
    return f"{run_id} — {emb}"


def _read_run_ids_from_specs(path: Path) -> list[str]:
    import tomllib

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    runs = raw.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError(f"'runs' must be a list in {path}")
    out: list[str] = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("run_id", "")).strip()
        if rid:
            out.append(rid)
    return out


def _archive_old_plots(plots_dir: Path) -> Path | None:
    if not plots_dir.exists():
        return None
    pngs = sorted(plots_dir.glob("*.png"))
    if not pngs:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = plots_dir / f"archive_{ts}"
    archive.mkdir(parents=True, exist_ok=True)
    for p in pngs:
        shutil.move(str(p), str(archive / p.name))
    return archive


def _run_dir(out_dir: Path, run_id: str) -> Path:
    return Path(out_dir) / "runs" / str(run_id)


def _run_coords_dir(out_dir: Path, run_id: str) -> Path:
    return _run_dir(out_dir, run_id) / "coords"


def _run_plots_dir(out_dir: Path, run_id: str) -> Path:
    return _run_dir(out_dir, run_id) / "plots"


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _load_pred_prob(*, out_dir: Path, run_id: str) -> pd.DataFrame | None:
    """
    Load per-probe predicted probabilities from the persisted logits memmap and return:
      DataFrame[row_idx, pred_prob]

    This keeps storage on-disk as logits, but enables probability plots.
    """
    probe_index = out_dir / "probe_index.csv"
    if not probe_index.exists():
        return None
    idx = pd.read_csv(probe_index, usecols=["row_idx"]).sort_values("row_idx")
    row_idx = idx["row_idx"].to_numpy(dtype=int)
    N = int(len(row_idx))
    if N <= 0:
        return None

    # New layout: per-run memmaps.
    mmap_path = out_dir / "runs" / run_id / "memmap" / f"{run_id}__logits__N{N}.fp32.mmap"
    if not mmap_path.exists():
        # Legacy fallback.
        mmap_path = out_dir / "memmap" / f"{run_id}__logits__N{N}.fp32.mmap"
        if not mmap_path.exists():
            return None

    logits = np.memmap(mmap_path, mode="r", dtype="float32", shape=(N,))
    probs = _sigmoid_np(np.asarray(logits, dtype=np.float32))
    return pd.DataFrame({"row_idx": row_idx, "pred_prob": probs})

@dataclass(frozen=True)
class PlotContext:
    df: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    extent: tuple[float, float, float, float]
    gridsize: int
    mincnt: int


PanelKind = Literal["density", "subset_density", "continuous_median"]


@dataclass(frozen=True)
class PanelSpec:
    kind: PanelKind
    label: str
    col: str | None = None
    color: str | None = None
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    colorbar_label: str | None = None
    show_background: bool = True
    mask_fn: Callable[[pd.DataFrame], np.ndarray] | None = None


def _make_grid_figure(
    *,
    grid: tuple[int, int],
    figsize: tuple[float, float],
    title: str,
    subtitle: str,
) -> tuple["Figure", list["Axes"]]:
    import matplotlib.pyplot as plt

    nrows, ncols = grid
    fig, ax_grid = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    # Keep the layout automatic, but reserve a fixed headroom for title/subtitle.
    # (Use `Any` here: matplotlib's layout engine API is not consistently captured by type stubs.)
    inline_subtitle = nrows == 1 and ncols >= 3 and bool(subtitle.strip())
    full_title = f"{title} — {subtitle}" if inline_subtitle else title
    title_y = 0.985 if inline_subtitle else 0.975
    title_fs = 13 if inline_subtitle else 15
    rect_top = 0.875 if inline_subtitle else 0.90

    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        try:
            cast(Any, engine).set(
                rect=(0.02, 0.02, 0.985, rect_top),
                w_pad=0.006,
                h_pad=0.006,
                wspace=0.01,
                hspace=0.01,
            )
        except Exception:
            pass

    fig.text(
        0.5,
        title_y,
        full_title,
        ha="center",
        va="top",
        fontsize=title_fs,
        fontweight="semibold",
    )
    if not inline_subtitle and subtitle.strip():
        fig.text(
            0.5,
            0.94,
            subtitle,
            ha="center",
            va="top",
            fontsize=12,
        )

    axes = [cast("Axes", ax) for ax in ax_grid.ravel().tolist()]
    return fig, axes


def _axes_label(ax: "Axes", text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="medium",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85},
    )


def _add_colorbar(ax: "Axes", mappable: Any, *, label: str) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Put the colorbar *inside* the axes so it doesn't change subplot sizes (important for equal aspect).
    cax = inset_axes(
        ax,
        width="3.3%",
        height="68%",
        loc="center right",
        # Keep the colorbar strictly inside the axes box so it can't spill into the neighbor subplot.
        bbox_to_anchor=(0.0, 0.0, 0.975, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.15,
    )

    # Ensure a solid background so any small misalignment doesn't reveal underlying plot.
    cax.set_facecolor("white")
    # Hide spines for a cleaner inset look.
    for spine in cax.spines.values():
        spine.set_visible(False)

    cb = plt.colorbar(mappable, cax=cax)
    # Reduce label padding (can use a negative value to move it closer)
    cb.set_label(label, labelpad=-6)
    cb.ax.tick_params(labelsize=7)

    # Add a small white bbox behind the label to handle overlap/misalignment gracefully.
    lbl = cb.ax.yaxis.get_label()
    lbl.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.6})


def _draw_underlay(ax: "Axes", ctx: PlotContext, *, alpha: float = 0.18) -> None:
    ax.hexbin(
        ctx.x,
        ctx.y,
        gridsize=int(ctx.gridsize),
        extent=ctx.extent,
        mincnt=1,
        alpha=float(alpha),
        cmap="Greys",
        bins="log",
        linewidths=0.0,
        zorder=0,
    )


def _draw_density(ax: "Axes", ctx: PlotContext) -> None:
    ax.hexbin(
        ctx.x,
        ctx.y,
        gridsize=int(ctx.gridsize),
        extent=ctx.extent,
        mincnt=1,
        alpha=1.0,
        cmap="Greys",
        bins="log",
        linewidths=0.0,
        zorder=0,
    )


def _draw_subset_density(
    ax: "Axes",
    ctx: PlotContext,
    *,
    mask: np.ndarray,
    color: str,
    show_background: bool,
) -> None:
    import matplotlib.colors as mcolors

    if show_background:
        _draw_underlay(ax, ctx)

    sub = ctx.df.loc[mask, ["x", "y"]].to_numpy(dtype=float)
    if sub.shape[0] == 0:
        return
    rgba0 = (1.0, 1.0, 1.0, 0.0)
    rgba1 = mcolors.to_rgba(color, 1.0)
    ax.hexbin(
        sub[:, 0],
        sub[:, 1],
        gridsize=int(ctx.gridsize),
        extent=ctx.extent,
        mincnt=int(ctx.mincnt),
        cmap=mcolors.LinearSegmentedColormap.from_list("overlay", [rgba0, rgba1]),
        bins="log",
        linewidths=0.0,
        alpha=0.95,
        zorder=1,
    )


def _draw_continuous_median(
    ax: "Axes",
    ctx: PlotContext,
    *,
    values: np.ndarray,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    colorbar_label: str,
    diverging_center: float | None = None,
) -> None:
    import matplotlib.colors as mcolors

    vals = np.asarray(values, dtype=float)
    norm = None
    if diverging_center is not None and np.isfinite(diverging_center):
        v = vals[np.isfinite(vals)]
        if v.size:
            vmax_auto = float(np.nanpercentile(np.abs(v - float(diverging_center)), 95))
            vmax_auto = max(vmax_auto, 1e-6)
            norm = mcolors.TwoSlopeNorm(
                vmin=float(diverging_center) - vmax_auto,
                vcenter=float(diverging_center),
                vmax=float(diverging_center) + vmax_auto,
            )

    hb = ax.hexbin(
        ctx.x,
        ctx.y,
        C=vals,
        reduce_C_function=cast(
            Callable[[np.ndarray | list[float]], float], _nanmedian_any
        ),
        gridsize=int(ctx.gridsize),
        extent=ctx.extent,
        mincnt=int(ctx.mincnt),
        cmap=str(cmap),
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        linewidths=0.0,
        zorder=1,
    )
    _add_colorbar(ax, hb, label=colorbar_label)


def _load_context(
    *,
    out_dir: Path,
    run_id: str,
    space: str,
    method: str,
    gridsize: int,
    mincnt: int,
) -> PlotContext:
    coords_path = _run_coords_dir(out_dir, run_id) / f"{run_id}__{space}__{method}.csv"
    meta_path = out_dir / "probe_metadata.csv"
    if not coords_path.exists():
        legacy = out_dir / "coords" / f"{run_id}__{space}__{method}.csv"
        if legacy.exists():
            coords_path = legacy
        else:
            raise FileNotFoundError(coords_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    coords = pd.read_csv(coords_path)
    meta = pd.read_csv(meta_path)
    if "chagas" not in meta.columns and "y_true" in meta.columns:
        meta = meta.rename(columns={"y_true": "chagas"})
    df = coords.merge(meta, on="row_idx", how="left")
    pred = _load_pred_prob(out_dir=out_dir, run_id=run_id)
    if pred is not None:
        df = df.merge(pred, on="row_idx", how="left")

    # Derived columns.
    if "abnormal_ecg" not in df.columns and "normal_ecg" in df.columns:
        df["abnormal_ecg"] = 1.0 - _prep_binary(df["normal_ecg"])
    if "timey" in df.columns:
        df["timey_present"] = (
            pd.to_numeric(df["timey"], errors="coerce").notna().astype(int)
        )

    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    df = df.loc[ok].copy()
    x = x[ok]
    y = y[ok]
    ext = _square_extent(_extent(x, y))
    return PlotContext(
        df=df,
        x=x,
        y=y,
        extent=ext,
        gridsize=int(gridsize),
        mincnt=int(mincnt),
    )


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df), index=df.index)


def _mask_equals(df: pd.DataFrame, col: str, value: object) -> np.ndarray:
    s = _safe_series(df, col)
    return (s.astype(str) == str(value)).to_numpy()


def _mask_binary(df: pd.DataFrame, col: str, value: int) -> np.ndarray:
    b = _prep_binary(_safe_series(df, col))
    return np.isfinite(b) & (b == float(int(value)))


def _render_figure(
    *,
    ctx: PlotContext,
    out_path: Path,
    title: str,
    subtitle: str,
    axis_prefix: str,
    grid: tuple[int, int],
    figsize: tuple[float, float],
    panels: list[PanelSpec],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = _make_grid_figure(
        grid=grid, figsize=figsize, title=title, subtitle=subtitle
    )
    nrows, ncols = grid

    for ax, p in zip(axes, panels, strict=False):
        if p.kind == "density":
            _draw_density(ax, ctx)
            _axes_label(ax, p.label)
            continue

        if p.kind == "subset_density":
            if p.mask_fn is not None:
                mask = np.asarray(p.mask_fn(ctx.df), dtype=bool)
            elif p.col is not None and p.col in ctx.df.columns:
                mask = _prep_binary(ctx.df[p.col]) == 1.0
            else:
                ax.axis("off")
                _axes_label(ax, f"{p.label}\n(missing)")
                continue
            _draw_subset_density(
                ax,
                ctx,
                mask=mask,
                color=str(p.color or "#4C78A8"),
                show_background=bool(p.show_background),
            )
            _axes_label(ax, p.label)
            continue

        if p.kind == "continuous_median":
            if p.col is None or p.col not in ctx.df.columns:
                ax.axis("off")
                _axes_label(ax, f"{p.label}\n(missing)")
                continue
            vals = pd.to_numeric(ctx.df[p.col], errors="coerce").to_numpy(dtype=float)
            _draw_continuous_median(
                ax,
                ctx,
                values=vals,
                cmap=str(p.cmap or "viridis"),
                vmin=p.vmin,
                vmax=p.vmax,
                colorbar_label=str(p.colorbar_label or p.col),
                diverging_center=0.0 if p.col == "delta_age" else None,
            )
            _axes_label(ax, p.label)
            continue

        raise ValueError(f"Unsupported panel kind: {p.kind}")

    # Apply shared limits/aspect *after* all hexbin calls. hexbin can autoscale axes, and if we
    # set limits/aspect before plotting it may get undone (most obvious with UMAP).
    for i, ax in enumerate(axes):
        r = i // ncols
        c = i % ncols
        # Shared axes: label only the outer axes to keep the figure clean.
        if r == nrows - 1:
            ax.set_xlabel(f"{axis_prefix}1", fontsize=10)
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel(f"{axis_prefix}2", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_xlim(ctx.extent[0], ctx.extent[1])
        ax.set_ylim(ctx.extent[2], ctx.extent[3])
        # Make every subplot box square; combined with square extents and equal aspect
        # this avoids UMAP panels looking "squished" by layout geometry.
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=9)
        ax.label_outer()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _default_figures() -> list[
    tuple[str, tuple[int, int], tuple[float, float], str, list[PanelSpec]]
]:
    # Returns a list of:
    #   (name, grid, figsize, subtitle, panels)
    return [
        (
            "multipanel_main",
            (3, 2),
            (7.6, 11.0),
            "Embedding geometry & ECG phenotype alignment",
            [
                PanelSpec("density", "Density (geometry)", show_background=False),
                PanelSpec("subset_density", "Chagas", col="chagas", color="#F58518"),
                PanelSpec("subset_density", "RBBB", col="RBBB", color="#4C78A8"),
                PanelSpec(
                    "subset_density",
                    "Abnormal ECG",
                    col="abnormal_ecg",
                    color="#54A24B",
                ),
                PanelSpec(
                    "continuous_median",
                    "Δage (median)",
                    col="delta_age",
                    cmap="RdBu_r",
                    colorbar_label="years",
                    show_background=False,
                ),
                PanelSpec(
                    "continuous_median",
                    "Predicted p(Chagas) (median)",
                    col="pred_prob",
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                    colorbar_label="p",
                    show_background=False,
                ),
            ],
        ),
        (
            "multipanel_conduction",
            (2, 2),
            (7.6, 7.6),
            "Conduction / rhythm alignment",
            [
                PanelSpec("subset_density", "RBBB", col="RBBB", color="#4C78A8"),
                PanelSpec("subset_density", "1dAVb", col="1dAVb", color="#9467BD"),
                PanelSpec("subset_density", "LBBB", col="LBBB", color="#54A24B"),
                PanelSpec("subset_density", "AF", col="AF", color="#F58518"),
            ],
        ),
        (
            "multipanel_outcome",
            (2, 2),
            (7.6, 7.6),
            "Clinical outcome alignment",
            [
                PanelSpec("density", "Density (geometry)", show_background=False),
                PanelSpec("subset_density", "Death", col="death", color="#D62728"),
                PanelSpec(
                    "continuous_median",
                    "Follow-up time (median)",
                    col="timey",
                    cmap="viridis",
                    colorbar_label="years",
                    show_background=False,
                ),
                PanelSpec(
                    "subset_density",
                    "Follow-up available",
                    col="timey_present",
                    color="#7F7F7F",
                ),
            ],
        ),
        (
            "sm_dataset_conditions",
            (1, 4),
            (14.8, 3.9),
            "Dataset conditions (domain shift)",
            [
                PanelSpec(
                    "subset_density",
                    "SAMITROP (+)",
                    color="#54A24B",
                    mask_fn=lambda df: _mask_equals(df, "dataset_source", "SAMITROP"),
                ),
                PanelSpec(
                    "subset_density",
                    "PTBXL (-)",
                    color="#F58518",
                    mask_fn=lambda df: _mask_equals(df, "dataset_source", "PTBXL"),
                ),
                PanelSpec(
                    "subset_density",
                    "CODE15 (+)",
                    color="#E45756",
                    mask_fn=lambda df: _mask_equals(df, "dataset_source", "CODE15")
                    & _mask_binary(df, "chagas", 1),
                ),
                PanelSpec(
                    "subset_density",
                    "CODE15 (-)",
                    color="#4C78A8",
                    mask_fn=lambda df: _mask_equals(df, "dataset_source", "CODE15")
                    & _mask_binary(df, "chagas", 0),
                ),
            ],
        ),
        (
            "sm_quality_overall",
            (1, 4),
            (14.8, 3.9),
            "Signal Quality",
            [
                PanelSpec(
                    "continuous_median",
                    "Template-match QC (median)",
                    col="qc_templatematch_bp",
                    cmap="viridis",
                    vmin=0.9,
                    vmax=1.0,
                    colorbar_label="qc_templatematch_bp",
                    show_background=False,
                ),
                PanelSpec(
                    "subset_density",
                    "Excellent",
                    color="#2CA02C",
                    mask_fn=lambda df: _mask_equals(df, "qc_zhao2018_bp", "Excellent"),
                ),
                PanelSpec(
                    "subset_density",
                    "Barely acceptable",
                    color="#F5A623",
                    mask_fn=lambda df: _mask_equals(
                        df, "qc_zhao2018_bp", "Barely acceptable"
                    ),
                ),
                PanelSpec(
                    "subset_density",
                    "Unacceptable",
                    color="#D62728",
                    mask_fn=lambda df: _mask_equals(
                        df, "qc_zhao2018_bp", "Unacceptable"
                    ),
                ),
            ],
        ),
        (
            "sm_chagas_x_rbbb",
            (1, 4),
            (14.8, 3.9),
            "Chagas × RBBB",
            [
                PanelSpec(
                    "subset_density",
                    "chagas=0, RBBB=0",
                    color="#4C78A8",
                    mask_fn=lambda df: _mask_binary(df, "chagas", 0)
                    & _mask_binary(df, "RBBB", 0),
                ),
                PanelSpec(
                    "subset_density",
                    "chagas=0, RBBB=1",
                    color="#9ECAE9",
                    mask_fn=lambda df: _mask_binary(df, "chagas", 0)
                    & _mask_binary(df, "RBBB", 1),
                ),
                PanelSpec(
                    "subset_density",
                    "chagas=1, RBBB=0",
                    color="#F58518",
                    mask_fn=lambda df: _mask_binary(df, "chagas", 1)
                    & _mask_binary(df, "RBBB", 0),
                ),
                PanelSpec(
                    "subset_density",
                    "chagas=1, RBBB=1",
                    color="#D62728",
                    mask_fn=lambda df: _mask_binary(df, "chagas", 1)
                    & _mask_binary(df, "RBBB", 1),
                ),
            ],
        ),
    ]


def _run_plotting(
    *,
    out_dir: Path,
    run_ids: list[str],
    spaces: list[str],
    methods: list[str],
    gridsize: int,
    mincnt: int,
    dpi: int,
    clean: bool,
) -> None:
    figures = _default_figures()

    for rid in run_ids:
        plots_dir = _run_plots_dir(out_dir, rid)
        plots_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            archive = _archive_old_plots(plots_dir)
            if archive is not None:
                print(f"Archived old plots to {archive}")
        for space in spaces:
            for method in methods:
                ctx = _load_context(
                    out_dir=out_dir,
                    run_id=rid,
                    space=space,
                    method=method,
                    gridsize=gridsize,
                    mincnt=mincnt,
                )
                title = _format_run_title(run_id=rid, space=space, method=method)
                axis_prefix = "UMAP" if str(method).lower() == "umap" else "PCA"
                for name, grid, figsize, subtitle, panels in figures:
                    out_path = plots_dir / f"{rid}__{space}__{method}__{name}.png"
                    _render_figure(
                        ctx=ctx,
                        out_path=out_path,
                        title=title,
                        subtitle=subtitle,
                        axis_prefix=axis_prefix,
                        grid=grid,
                        figsize=figsize,
                        panels=panels,
                        dpi=dpi,
                    )
                    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hex-tiling multipanel plots for the probe analysis (one script, no KDE, no scatter)."
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("analysis/embeddings_probe")
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_specs", type=Path, help="TOML run registry.")
    group.add_argument(
        "--run_id", action="append", default=[], help="Run id (repeatable)."
    )

    parser.add_argument(
        "--spaces", type=str, default="enc", help="Comma-separated: enc,proj"
    )
    parser.add_argument(
        "--methods", type=str, default="umap,pca", help="Comma-separated: umap,pca"
    )
    parser.add_argument(
        "--gridsize", type=int, default=30, help="Hex gridsize (lower => bigger hexes)."
    )
    parser.add_argument(
        "--mincnt", type=int, default=3, help="Minimum points per colored hex."
    )
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Archive existing PNGs in <out_dir>/plots/ first.",
    )
    args = parser.parse_args()

    if args.run_specs:
        run_ids = _read_run_ids_from_specs(args.run_specs)
    else:
        run_ids = [str(x).strip() for x in args.run_id if str(x).strip()]
    if not run_ids:
        raise ValueError("No run ids provided.")

    spaces = [s.strip() for s in str(args.spaces).split(",") if s.strip()]
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    _run_plotting(
        out_dir=args.out_dir,
        run_ids=run_ids,
        spaces=spaces,
        methods=methods,
        gridsize=int(args.gridsize),
        mincnt=int(args.mincnt),
        dpi=int(args.dpi),
        clean=bool(args.clean),
    )


if __name__ == "__main__":
    main()
