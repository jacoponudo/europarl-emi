"""REVIEW ONLY: compare the measurement variants with the baseline.

    uv run python notebooks/review/sensitivity/compare.py

For every variant computed so far (v1_mpnet, v2_word2vec, and any --sample run; --smoke runs are ignored):
  - correlation of speech-level EMI with the baseline (original-English sample);
  - change in EMI per decade, 1996-2019, without and with speaker fixed effects;
  - change between 1996-1998 and 2017-2019, original-English sample and sample including translations.
Output: notebooks/review/sensitivity/sensitivity.csv and sensitivity_trends.(pdf|png)
"""

import json
import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from europarl_emi import config as cfg
from europarl_emi.figures import GREY, MM, MUTED, OCHRE, PEACH, SAGE, set_style, yearly, zero_line
from europarl_emi.step08_models import ANNOTATED, fit_trend, load_main

OUT_ROOT = cfg.ROOT / "data" / "review" / "sensitivity"
HERE = cfg.ROOT / "notebooks" / "review" / "sensitivity"
TABLE = HERE / "sensitivity.csv"
FIGURE_DIR = HERE
EARLY, LATE = (1996, 1998), (2017, 2019)


def load_variants():
    """Baseline plus every computed variant. For a sampled variant, the baseline is also
    evaluated on exactly the same speeches ('baseline on <variant>'), so the two are comparable."""
    baseline = pd.read_parquet(cfg.PROCESSED / "05_speech_emi.parquet", columns=["guid", "emi"])
    out = {"baseline": baseline}
    found = sorted(p.parent.name for p in OUT_ROOT.glob("*/speech_emi.parquet")
                   if (p.parent / "settings.json").exists() and not p.parent.name.startswith("_"))
    if not found:
        print("  no variant computed yet (run v1_mpnet.py and/or v2_word2vec.py)")
    for folder in found:
        settings = json.loads((OUT_ROOT / folder / "settings.json").read_text())
        print(f"  {folder}: model {settings['model']}, {settings['chunk_max']}-word chunks, "
              f"header {'removed' if settings['strip_header'] else 'kept'}")
    for folder in found:
        emi = pd.read_parquet(OUT_ROOT / folder / "speech_emi.parquet", columns=["guid", "emi"])
        if "_sample" in folder:
            out[f"baseline on {folder}"] = baseline[baseline["guid"].isin(set(emi["guid"]))]
        out[folder] = emi
    return out


def change(df):
    early = df.loc[df["year"].between(*EARLY), "emi"].mean()
    late = df.loc[df["year"].between(*LATE), "emi"].mean()
    return late - early


def main():
    variants = load_variants()
    base = load_main(period=None).drop(columns=["emi"])
    oral = pd.read_parquet(ANNOTATED, columns=["guid", "year", "role", "written"])
    oral = oral.loc[(oral["role"] == "mep") & ~oral["written"], ["guid", "year"]]
    baseline_emi = variants["baseline"].rename(columns={"emi": "emi_baseline"})

    rows, series = [], {}
    for name, emi in variants.items():
        main_v = base.merge(emi, on="guid", how="inner")
        core = main_v[main_v["year"].between(1996, 2019)]
        oral_v = oral.merge(emi, on="guid", how="inner")
        corr = core.merge(baseline_emi, on="guid")[["emi", "emi_baseline"]].corr().iloc[0, 1]
        plain = fit_trend(core, [], name)
        fe = fit_trend(core, ["mep_id"], name)
        rows.append({
            "variant": name,
            "speeches_1996_2019": len(core),
            "corr_with_baseline": round(corr, 3),
            "trend_per_decade": round(plain["coef"], 3),
            "trend_ci": f"[{plain['ci_low']:.3f}, {plain['ci_high']:.3f}]",
            "trend_speaker_fe": round(fe["coef"], 3),
            "trend_speaker_fe_ci": f"[{fe['ci_low']:.3f}, {fe['ci_high']:.3f}]",
            "change_original_english": round(change(core), 3),
            "change_incl_translations": round(change(oral_v[oral_v["year"].between(1996, 2019)]), 3),
        })
        series[name] = (yearly(core, "emi"), yearly(oral_v[oral_v["year"].between(1996, 2019)], "emi"))

    table = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))

    names = [n for n in series if not n.startswith("baseline")]
    if not names:
        return
    set_style()
    fig, axes = plt.subplots(2, len(names), figsize=(max(60 * len(names), 120) * MM, 105 * MM), sharex=True, sharey=True,
                             squeeze=False)
    for col, name in enumerate(names):
        for row, (label, band) in enumerate((("Delivered in English", PEACH), ("Including translations", SAGE))):
            ax = axes[row, col]
            zero_line(ax)
            reference = f"baseline on {name}" if f"baseline on {name}" in series else "baseline"
            b = series[reference][row]
            s = series[name][row]
            ax.plot(b.index, b["mean"], color=GREY, ls="--", linewidth=0.9, label=reference)
            ax.fill_between(s.index, s["lo"], s["hi"], color=band, alpha=0.5, linewidth=0)
            ax.plot(s.index, s["mean"], color=OCHRE, marker="o", markersize=2.5, linewidth=1, label=name)
            ax.set_title(f"{name}: {label}", loc="left", fontsize=7)
            if col == 0:
                ax.set_ylabel("EMI")
            if row == 1:
                ax.set_xlabel("Year")
    axes[0, 0].legend(loc="lower left", fontsize=6)
    fig.text(0.01, -0.02, "Bands: 95% percentile bootstrap intervals. Dashed: baseline (header kept, "
             "150-word chunks, MiniLM).", fontsize=6, color=MUTED)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"sensitivity_trends.{ext}")
    print(f"Saved {TABLE.relative_to(cfg.ROOT)} and {(FIGURE_DIR / 'sensitivity_trends.png').relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
