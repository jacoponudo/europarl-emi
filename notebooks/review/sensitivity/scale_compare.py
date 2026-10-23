"""REVIEW ONLY. Compare the trend of every variant with the baseline on the same scale.

EMI is in units of the chunk-level standard deviation, which differs between variants
(50-word chunks are noisier than 150-word chunks). Here every trend is also divided by the
standard deviation of speech-level EMI of that variant, on exactly the same speeches,
so that the numbers are comparable. Also prints the correlation of the yearly means.

Figure (scale_compare.png/pdf):
  a  yearly mean EMI of every variant, in standard deviations of its speech-level EMI;
  b  trend per decade in the same units, without and with speaker fixed effects (95% CI, clustered by speaker).

    uv run python notebooks/review/sensitivity/scale_compare.py              # full runs only
    uv run python notebooks/review/sensitivity/scale_compare.py --samples    # include --sample runs

Output: printed, and saved to notebooks/review/sensitivity/scale_compare.csv and scale_compare.(png|pdf)
"""

import argparse
import json
import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from europarl_emi import config as cfg
from europarl_emi.figures import (ASH, DOUBLE, GREY, INK, MM, MUTED, OCHRE, PEACH, PLUM, SAGE, SLATE, WINE,
                                  set_style, yearly, zero_line)
from europarl_emi.step08_models import fit_trend, load_main

R = cfg.ROOT / "data" / "review" / "sensitivity"
HERE = cfg.ROOT / "notebooks" / "review" / "sensitivity"
OUT = HERE / "scale_compare.csv"
STYLES = [  # baseline first, then the variants in alphabetical order
    {"color": INK, "ls": "-", "marker": "o"},
    {"color": OCHRE, "ls": "-", "marker": "s"},
    {"color": WINE, "ls": "--", "marker": "^"},
    {"color": SAGE, "ls": "-.", "marker": "D"},
    {"color": GREY, "ls": ":", "marker": "v"},
    {"color": PLUM, "ls": "--", "marker": "P"},
    {"color": SLATE, "ls": "-.", "marker": "X"},
    {"color": ASH, "ls": ":", "marker": "h"},
]


def label(name, folder):
    """Readable label from settings.json: model, chunk size, header, scoring."""
    if name == "baseline":
        return "Paper: MiniLM, 150 words, header kept"
    s = json.loads((folder / "settings.json").read_text())
    model = {"sentence-transformers/all-mpnet-base-v2": "mpnet",
             "word2vec-google-news-300": "word2vec"}.get(s["model"], s["model"].split("/")[-1])
    if "percentile" in s.get("representation", ""):
        model += " (80th pct)"
    elif model == "word2vec":
        model += " (mean)"
    header = "header removed" if s["strip_header"] else "header kept"
    return f"{name.split('_')[0]}: {model}, {s['chunk_max']} words, {header}"


def plot(series, trends, labels):
    set_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 78 * MM), gridspec_kw={"width_ratios": [1.5, 1]})
    names = list(series)
    zero_line(a)
    for k, name in enumerate(names):
        st, s = STYLES[k % len(STYLES)], series[name]
        if name == "baseline":
            a.fill_between(s.index, s["lo"], s["hi"], color=PEACH, alpha=0.45, linewidth=0, zorder=1)
        a.plot(s.index, s["mean"], color=st["color"], ls=st["ls"], marker=st["marker"], markersize=2.5,
               linewidth=1.3 if name == "baseline" else 0.9, label=labels[name], zorder=3 if name == "baseline" else 2)
    a.set_xlabel("Year")
    a.set_ylabel("EMI (SD of speech-level EMI)")
    a.set_title("a  Yearly mean, same speeches, same scale", loc="left", fontweight="bold")
    a.legend(loc="lower left", fontsize=5.5, frameon=False)

    y = np.arange(len(names))[::-1]
    zero_line(b, axis="v")
    for k, name in enumerate(names):
        st = STYLES[k % len(STYLES)]
        for dy, spec, face in ((0.15, "none", st["color"]), (-0.15, "speaker", "white")):
            t = trends[(name, spec)]
            b.errorbar(t["coef"], y[k] + dy, xerr=[[t["coef"] - t["ci_low"]], [t["ci_high"] - t["coef"]]],
                       fmt=st["marker"], color=st["color"], mfc=face, markersize=3.5, capsize=1.5, linewidth=0.8)
    b.set_yticks(y, [labels[n].split(":")[0] for n in names])
    b.set_xlabel("Trend per decade (SD of speech-level EMI)")
    b.set_title("b  Trend, 1996–2019", loc="left", fontweight="bold")
    b.plot([], [], "o", color=MUTED, markersize=3.5, label="No fixed effects")
    b.plot([], [], "o", color=MUTED, mfc="white", markersize=3.5, label="Speaker fixed effects")
    b.legend(loc="upper right", fontsize=5.5, frameon=False)
    fig.text(0.01, -0.03, "Original-English oral speeches by MEPs, 1996–2019, identical in every variant. EMI of each "
             "variant centred and divided by its speech-level SD.\nBand: 95% percentile bootstrap for the paper "
             "measure. Bars: 95% CI, standard errors clustered by speaker.", fontsize=5.5, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"scale_compare.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {(HERE / 'scale_compare.png').relative_to(cfg.ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true", help="also include runs made with --sample")
    args = parser.parse_args()

    files = {"baseline": cfg.PROCESSED / "05_speech_emi.parquet"}
    for p in sorted(R.glob("*/speech_emi.parquet")):
        name = p.parent.name
        if (p.parent / "settings.json").exists() and not name.startswith("_") and (args.samples or "_sample" not in name):
            files[name] = p
    print("Variants:", ", ".join(files))

    main_sample = load_main(period=None).drop(columns=["emi"])
    main_sample = main_sample[main_sample["year"].between(1996, 2019)]
    emi = {k: pd.read_parquet(p, columns=["guid", "emi"]) for k, p in files.items()}
    same = set(main_sample["guid"]).intersection(*(set(v["guid"]) for v in emi.values()))

    rows, yearly_means, series, trends = [], {}, {}, {}
    for name, v in emi.items():
        d = main_sample.merge(v, on="guid")
        d = d[d["guid"].isin(same)]
        sd = d["emi"].std()
        z = d.assign(emi=(d["emi"] - d["emi"].mean()) / sd)
        series[name] = yearly(z, "emi")
        trends[(name, "none")] = fit_trend(z, [], name)
        trends[(name, "speaker")] = fit_trend(z, ["mep_id"], name)
        slope = np.polyfit(d["year"].astype(float), d["emi"], 1)[0] * 10
        change = (d.loc[d["year"].between(2017, 2019), "emi"].mean()
                  - d.loc[d["year"].between(1996, 1998), "emi"].mean())
        yearly_means[name] = d.groupby("year")["emi"].mean()
        rows.append({"variant": name, "speeches": len(d), "sd_speech_emi": sd,
                     "trend_per_decade": slope, "trend_in_sd": slope / sd, "change_in_sd": change / sd,
                     "trend_in_sd_speaker_fe": trends[(name, "speaker")]["coef"]})

    table = pd.DataFrame(rows).round(3)
    corr = pd.DataFrame(yearly_means).corr()
    table["corr_yearly_with_baseline"] = table["variant"].map(corr["baseline"]).round(3)
    pd.set_option("display.width", 200)
    print(f"\nSame {len(same):,} speeches (original English, oral, MEPs, 1996-2019) in every variant\n")
    print(table.to_string(index=False))
    print("\nCorrelation of yearly means:")
    print(corr.round(3).to_string())
    table.to_csv(OUT, index=False)
    print(f"\nSaved {OUT.relative_to(cfg.ROOT)}")
    plot(series, trends, {n: label(n, files[n].parent) for n in files})


if __name__ == "__main__":
    main()
