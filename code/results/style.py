"""Shared figure style for the paper figures: Nature Human Behaviour look, ECL orange palette, PDF output.

    from results.style import PALETTE, C, apply_style, panel_label, mm, save   # with code/ on sys.path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Project palette: the "orange" colour mode of the Europa Component Library (ECL v5, CSS class
# ecl-color-mode--orange, @ecl/preset-ec). Names are the ECL custom properties.
PALETTE = {
    "orange": "#e87e1d",        # --cm-border-high
    "orange-light": "#ff8a20",  # --cm-on-surface-1 / --cm-border
    "brown": "#8c4c12",         # --cm-on-surface
    "teal-dark": "#0a3f4d",     # --cm-surface-variant-1
    "teal": "#69d2c6",          # --cm-on-surface-2
    "surface": "#ffa14d",       # --cm-surface-0
    "surface-medium": "#ffc998",  # --cm-surface-medium-0 / --cm-border-low
    "surface-low": "#ffdbba",   # --cm-surface-low-0
    "surface-lowest": "#fff3e9",  # --cm-surface-lowest (before the 50% mix)
}
# Categorical order for series (fixed, never cycled): orange, brown, dark teal, teal.
# Checked with the dataviz validator: adjacent pairs distinct also for colour-blind readers (worst dE 14 protan).
# Teal (#69d2c6) and orange have low contrast on white (1.8:1, 2.8:1): every series also has a legend entry,
# and the numbers are written to results/tables/.
C = [PALETTE["orange"], PALETTE["brown"], PALETTE["teal-dark"], PALETTE["teal"]]

# Groups of speeches (column `group` of translation.unified.load): (key, legend label, colour).
# The same group keeps the same colour in every figure.
ORIGINAL = ("original", "Delivered in English", PALETTE["orange"])
HUMAN = ("human", "Translated by human translators", PALETTE["brown"])
ETRANSLATION = ("etranslation", "Translated by eTranslation", PALETTE["teal-dark"])
ALL = ("all", "All speeches", PALETTE["teal"])  # aggregate: dashed, open rings


# The two components of EMI (evidence_z, intuition_z), when drawn without groups: (key, legend label, colour)
EVIDENCE = ("evidence_z", "Evidence", PALETTE["teal-dark"])
INTUITION = ("intuition_z", "Intuition", PALETTE["orange"])


def series(without_human: bool = False) -> list[tuple[str, str, str]]:
    """The groups drawn, in order; --without-human leaves out the Parliament's human translations (to 2012)."""
    return [ORIGINAL, ETRANSLATION] if without_human else [ORIGINAL, HUMAN, ETRANSLATION]

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
    """One PDF per figure (vector, editable text): results/figures/paper/<name>.pdf."""
    path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path}")
