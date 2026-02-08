from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path("thesis/figures/generated/background_pipeline_overview.png")


def box(ax, x, y, w, h, text, fc="#f4f6fb", ec="#2b2b2b", lw=1.2, fs=10.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def center_right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def center_left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def center_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def center_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def arrow(ax, p1, p2, color="#222", lw=1.5, rad=0.0):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Columns: preprocessing -> tracks -> stage chain
    pre_w, pre_h = 0.14, 0.10
    trk_w, trk_h = 0.16, 0.11
    st_w, st_h = 0.24, 0.09

    x_pre, x_trk, x_st = 0.06, 0.29, 0.56

    # Preprocessing nodes
    pre_bp = box(ax, x_pre, 0.73, pre_w, pre_h, "Preprocessing\nbp", fc="#e8f2ff")
    pre_bpsc = box(ax, x_pre, 0.57, pre_w, pre_h, "Preprocessing\nbp-sc", fc="#e8f2ff")
    pre_bpscn = box(ax, x_pre, 0.41, pre_w, pre_h, "Preprocessing\nbp-sc-norm", fc="#e8f2ff")

    # Track nodes
    t1 = box(ax, x_trk, 0.70, trk_w, trk_h, "Track 1\nClassification", fc="#fff2db")
    t2 = box(ax, x_trk, 0.52, trk_w, trk_h, "Track 2\nPretraining", fc="#eaf7ea")
    t3 = box(ax, x_trk, 0.34, trk_w, trk_h, "Track 3\nLinear Probe", fc="#f7ebfb")

    # Stage chain
    s1 = box(ax, x_st, 0.73, st_w, st_h, "Stage 1\nTask Utility", fc="#fffaf0")
    s2 = box(ax, x_st, 0.59, st_w, st_h, "Stage 2\nEmbedding Health", fc="#fffaf0")
    s3 = box(ax, x_st, 0.45, st_w, st_h, "Stage 3\nCross-Model Consistency", fc="#fffaf0")
    s4 = box(ax, x_st, 0.31, st_w, st_h, "Stage 4\nPlausibility with XAI", fc="#fffaf0")

    # Preprocessing -> Track 1 and Track 2 (for each regime)
    for p in (pre_bp, pre_bpsc, pre_bpscn):
        arrow(ax, center_right(p), center_left(t1), rad=0.0)
        arrow(ax, center_right(p), center_left(t2), rad=0.0)

    # Track 2 -> Track 3
    arrow(ax, center_bottom(t2), center_top(t3), color="#4d4d4d")

    # Track 1 and Track 3 -> Stage 1
    arrow(ax, center_right(t1), center_left(s1), rad=0.05)
    arrow(ax, center_right(t3), center_left(s1), rad=-0.22)

    # Stage chain
    arrow(ax, center_bottom(s1), center_top(s2))
    arrow(ax, center_bottom(s2), center_top(s3))
    arrow(ax, center_bottom(s3), center_top(s4))

    ax.text(0.5, 0.95, "Study Design Overview", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.91, "Preprocessing regimes feed Track 1/2; Track 2 pretraining feeds Track 3; stages follow sequentially.",
            ha="center", va="center", fontsize=10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
