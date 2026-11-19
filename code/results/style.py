"""Shared figure style for the paper figures: Nature Human Behaviour look and the project palette.

    from results.style import PALETTE, C, apply_style, panel_label, mm, save   # with code/ on sys.path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Project palette, in order (warm to cool)
PALETTE = {
    "Cherry Rose": "#b7094c",
    "Dark Raspberry": "#a01a58",
    "Royal Plum": "#892b64",
    "Velvet Purple": "#723c70",
    "Dusty Grape": "#5c4d7d",
    "Dusk Blue": "#455e89",
    "Rich Cerulean": "#2e6f95",
    "Cerulean": "#1780a1",
    "Pacific Cyan": "#0091ad",
}
C = list(PALETTE.values())

INK = "#1a1a1a"      # text
MUTED = "#555555"    # axes, ticks
GRID = "#e6e6e6"     # gridlines
BAND = "#f2f2f2"     # background shading

# Nature figure widths (mm): single column 89, 1.5 column 120, double column 183
SINGLE, ONE_HALF, DOUBLE = 89, 120, 183


def mm(x: float) -> float:
    return x / 25.4


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",  # editable text
        "savefig.dpi": 600, "figure.dpi": 150,
    })


def panel_label(ax, letter: str, x: float = -0.07, y: float = 1.04) -> None:
    """Bold lowercase panel letter, top left, Nature style."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right", color=INK)


def ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(path.with_suffix(ext), bbox_inches="tight", dpi=600)
    print(f"figure: {path.with_suffix('.pdf')} / .png")
