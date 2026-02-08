from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    leads = [
        ("I", 0),
        ("-aVR", 30),
        ("II", 60),
        ("aVF", 90),
        ("III", 120),
        ("-aVL", 150),
        ("-I", 180),
        ("aVR", -150),
        ("-II", -120),
        ("-aVF", -90),
        ("-III", -60),
        ("aVL", -30),
    ]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    for name, deg in leads:
        theta = np.deg2rad(deg % 360)
        ax.plot([theta, theta], [0, 1.0], linewidth=1.5, color="#2f2f2f", zorder=2)

    for name, deg in leads:
        theta = np.deg2rad(deg % 360)
        ax.text(
            theta,
            1.11,
            f"{name}\n{deg}°",
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=6,
        )

    # Anatomical orientation hints (didactic only; not electrode placement coordinates).
    anatomical = [
        ("Left arm", 10, 1.30),               # slightly below Lead I axis for readability
        ("Right arm", 170, 1.30),             # slightly below -I axis for readability
        ("Left leg / inferior", 94, 1.30),    # near aVF (inferior axis)
    ]
    for txt, deg, rad in anatomical:
        theta = np.deg2rad(deg % 360)
        ax.text(
            theta,
            rad,
            txt,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#303030",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f5f5f5", edgecolor="#c9c9c9", alpha=0.95),
            zorder=6,
        )

    # Clean center marker: no emoji/symbol overlay.
    ax.scatter([0], [0], s=300, color="white", edgecolor="none", zorder=7)
    ax.scatter([0], [0], s=150, color="#b30000", edgecolor="black", linewidth=0.9, zorder=8)
    ax.text(0, 0.22, "Heart", ha="center", va="center", fontsize=10, fontweight="bold", color="#111111", zorder=9)

    ax.set_rticks([])
    ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{d}°" for d in range(0, 360, 30)], fontsize=8)
    ax.grid(True, linewidth=0.8, alpha=0.5)
    ax.set_rmax(1.34)
    ax.set_title("Cabrera Circle (Frontal Plane / Hexaxial Reference)", pad=18, fontsize=12)

    plt.tight_layout()
    out = Path("thesis/figures/generated/datasets_overview")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"cabrera_circle.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved", out / "cabrera_circle.png")


if __name__ == "__main__":
    main()
