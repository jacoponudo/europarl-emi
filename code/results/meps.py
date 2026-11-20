"""Who the speaker is: country and political group of the MEP of every speech, and EMI by category.
Used by fig4.py (country) and fig5.py (political group).

    from results.meps import speeches, category_means, small_multiples      # with code/ on sys.path

Country  from data/raw/inherited/meps/mep_database_1994_present.csv (columns mep_id, term, country): the row of
         the speaker (person_id) in the parliamentary term of the speech; if missing, the most frequent country
         of that MEP in the other terms. Values may be ISO codes, names or URIs ending with an ISO code; all
         are mapped to the ISO 3166 alpha-3 code. Speeches from a country before it joined the EU are flagged
         (country_check = "before accession") and left out of Figure 4.
Group    the European political group at the time of the speech, from the acronym in the header
         ("Syed Kamall (PPE-DE). - ..."), as in the paper's old step 7; if the header has none (corpus from 2020,
         some translated texts), the political_group of the database for that MEP and term. Groups are joined
         into families that are stable over time (FAMILIES, FAMILY_WORDS).
EMI by category: mean EMI of the speeches in each category and year, with 95% percentile bootstrap
         intervals over MEPs (all speeches of a resampled MEP enter together; a speech without person_id is
         its own cluster), B = 2000, seed 42.
"""

from __future__ import annotations

import html
from pathlib import Path
import re

import numpy as np
import pandas as pd

import config as cfg
from results.style import INK, MUTED

# ---------------------------------------------------------------- countries

# ISO alpha-3: (alpha-2, English name, year of EU accession)
COUNTRIES = {
    "BEL": ("BE", "Belgium", 1958), "DEU": ("DE", "Germany", 1958), "FRA": ("FR", "France", 1958),
    "ITA": ("IT", "Italy", 1958), "LUX": ("LU", "Luxembourg", 1958), "NLD": ("NL", "Netherlands", 1958),
    "DNK": ("DK", "Denmark", 1973), "IRL": ("IE", "Ireland", 1973), "GBR": ("GB", "United Kingdom", 1973),
    "GRC": ("GR", "Greece", 1981), "ESP": ("ES", "Spain", 1986), "PRT": ("PT", "Portugal", 1986),
    "AUT": ("AT", "Austria", 1995), "FIN": ("FI", "Finland", 1995), "SWE": ("SE", "Sweden", 1995),
    "CYP": ("CY", "Cyprus", 2004), "CZE": ("CZ", "Czechia", 2004), "EST": ("EE", "Estonia", 2004),
    "HUN": ("HU", "Hungary", 2004), "LVA": ("LV", "Latvia", 2004), "LTU": ("LT", "Lithuania", 2004),
    "MLT": ("MT", "Malta", 2004), "POL": ("PL", "Poland", 2004), "SVK": ("SK", "Slovakia", 2004),
    "SVN": ("SI", "Slovenia", 2004), "BGR": ("BG", "Bulgaria", 2007), "ROU": ("RO", "Romania", 2007),
    "HRV": ("HR", "Croatia", 2013),
}
_ALIASES = {"UK": "GBR", "EL": "GRC", "CZECH REPUBLIC": "CZE", "GREAT BRITAIN": "GBR"}
_TO_ISO3 = {**{k: k for k in COUNTRIES}, **{v[0]: k for k, v in COUNTRIES.items()},
            **{v[1].upper(): k for k, v in COUNTRIES.items()}, **_ALIASES}


def iso3(value) -> str | None:
    """ISO alpha-3 code of a country written as alpha-3, alpha-2, English name or URI ending with a code."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip().rstrip("/").split("/")[-1].split("#")[-1].strip().upper()
    return _TO_ISO3.get(v)


# ---------------------------------------------------------------- political groups

# acronym (upper case, dashes normalised) -> family
FAMILIES = {
    "PPE": "EPP", "PPE-DE": "EPP", "EPP": "EPP", "EPP-ED": "EPP",
    "PSE": "S&D", "PES": "S&D", "S&D": "S&D", "S-D": "S&D", "SPE": "S&D",
    "ELDR": "Liberals", "ALDE": "Liberals", "ADLE": "Liberals", "RENEW": "Liberals", "RE": "Liberals",
    "V": "Greens/EFA", "VERTS/ALE": "Greens/EFA", "GREENS/EFA": "Greens/EFA", "GREENS/ALE": "Greens/EFA",
    "GUE/NGL": "Left", "GUE": "Left", "LEFT": "Left", "THE LEFT": "Left",
    "UPE": "Conservatives", "UEN": "Conservatives", "ECR": "Conservatives",
    "EDN": "Eurosceptics", "I-EDN": "Eurosceptics", "EDD": "Eurosceptics", "IND/DEM": "Eurosceptics",
    "EFD": "Eurosceptics", "EFDD": "Eurosceptics", "ENF": "Eurosceptics", "ID": "Eurosceptics",
    "ITS": "Eurosceptics", "PFE": "Eurosceptics", "ESN": "Eurosceptics",
    "NI": "Non-attached", "TDI": "Non-attached",
    "ARE": "Other",
}
# full names of groups (database), checked in this order, lower case
FAMILY_WORDS = [
    ("non-attached", "Non-attached"), ("non-inscrits", "Non-attached"), ("technical group", "Non-attached"),
    ("people's party", "EPP"), ("christian democrat", "EPP"),
    ("socialists and democrats", "S&D"), ("socialist", "S&D"),
    ("renew", "Liberals"), ("liberal", "Liberals"),
    ("left", "Left"),  # before "green": "European United Left - Nordic Green Left"
    ("greens", "Greens/EFA"), ("free alliance", "Greens/EFA"),
    ("conservatives and reformists", "Conservatives"), ("union for europe", "Conservatives"),
    ("identity and democracy", "Eurosceptics"), ("nations and freedom", "Eurosceptics"),
    ("freedom and direct democracy", "Eurosceptics"), ("freedom and democracy", "Eurosceptics"),
    ("independence/democracy", "Eurosceptics"), ("democracies and diversities", "Eurosceptics"),
    ("patriots", "Eurosceptics"), ("sovereign nations", "Eurosceptics"), ("europe of nations", "Eurosceptics"),
    ("identity, tradition", "Eurosceptics"), ("radical alliance", "Other"),
]
FAMILY_ORDER = ["Left", "Greens/EFA", "S&D", "Liberals", "EPP", "Conservatives", "Eurosceptics",
                "Non-attached", "Other"]
_PAREN = re.compile(r"\(([^()]{1,25})\)")


def _key(raw: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‐‑–−—]", "-", html.unescape(raw))).strip().upper()


def group_from_header(header) -> str | None:
    """Last recognised group acronym in brackets in the header (skips e.g. '(The Earl of)')."""
    if not isinstance(header, str):
        return None
    found = _PAREN.findall(header)
    m = re.search(r"on behalf of the (.{1,25}?) group", header, re.IGNORECASE)
    if m:
        found.append(m.group(1))
    for raw in reversed(found):
        if _key(raw) in FAMILIES:
            return _key(raw)
    return None


def family_of(group) -> str | None:
    """Family of a group written as an acronym, a full name or a URI ending with an acronym."""
    if not isinstance(group, str) or not group.strip():
        return None
    k = _key(group.rstrip("/").split("/")[-1])
    if k in FAMILIES:
        return FAMILIES[k]
    low = group.lower()
    for word, fam in FAMILY_WORDS:
        if word in low:
            return fam
    return None


# ---------------------------------------------------------------- attributes of each speech

def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.extract(r"(\d+)", expand=False)


def attributes(uids: pd.Series) -> pd.DataFrame:
    """uid -> person_id, country (ISO3), country_name, country_check, group_raw, family, family_source."""
    u = pd.read_parquet(cfg.UNIFIED, columns=["uid", "year", "term", "person_id", "header"])
    u = u[u.uid.isin(set(uids))].copy()
    u["person_id"] = _norm_id(u.person_id)
    # plain numpy floats (NaN for missing): nullable Int64 / NA would break comparisons and the term join
    u["year"] = pd.to_numeric(u.year, errors="coerce").astype("float64")
    u["term"] = pd.to_numeric(u.term.astype("string").str.extract(r"(\d+)", expand=False),
                              errors="coerce").astype("float64")

    db = pd.read_csv(cfg.MEP_DB, dtype=str)
    print(f"{cfg.MEP_DB.name}: {len(db):,} rows, columns {list(db.columns)}")
    db = pd.DataFrame({"person_id": _norm_id(db["mep_id"]),
                       "term": pd.to_numeric(db["term"], errors="coerce").astype("float64"),
                       "country_db": db["country"].map(iso3) if "country" in db else None,
                       "group_db": db["political_group"] if "political_group" in db else None})
    db = db.dropna(subset=["person_id"])
    bad = db.country_db.isna().mean()
    print(f"database: {db.person_id.nunique():,} MEPs; country not recognised in {bad:.1%} of rows")
    by_term = db.dropna(subset=["term"]).drop_duplicates(["person_id", "term"]).set_index(["person_id", "term"])
    any_term = db.dropna(subset=["country_db"]).groupby("person_id").country_db.agg(lambda s: s.mode().iat[0])

    idx = pd.MultiIndex.from_arrays([u.person_id, u.term])
    u["country"] = by_term.country_db.reindex(idx).to_numpy()
    u["country"] = u.country.fillna(u.person_id.map(any_term))
    known = u.country.map(lambda c: isinstance(c, str) and c in COUNTRIES).to_numpy(dtype=bool)
    u["country_name"] = u.country.map(lambda c: COUNTRIES[c][1] if isinstance(c, str) and c in COUNTRIES else None)
    joined = u.country.map(lambda c: COUNTRIES[c][2] if isinstance(c, str) and c in COUNTRIES else np.nan)
    early = u.year.to_numpy(dtype=float) < joined.to_numpy(dtype=float)  # NaN -> False
    u["country_check"] = np.where(~known, "missing", np.where(early, "before accession", "ok"))

    u["group_raw"] = u.header.map(group_from_header)
    u["family"] = u.group_raw.map(FAMILIES)
    u["family_source"] = np.where(u.family.notna(), "header", None)
    group_db = by_term.group_db.reindex(idx).to_numpy()
    fam_db = pd.Series(group_db, index=u.index).map(family_of)
    both = u.family.notna() & fam_db.notna()
    if both.any():
        print(f"family from header and from database agree on {u.family[both].eq(fam_db[both]).mean():.1%} "
              f"of {both.sum():,} speeches with both")
    fill = u.family.isna() & fam_db.notna()
    u.loc[fill, "family"] = fam_db[fill]
    u.loc[fill, "family_source"] = "database"
    u["family_source"] = u.family_source.fillna("missing")
    unknown = pd.Series(group_db, index=u.index)[u.family.isna()].value_counts().head(10)
    if len(unknown):
        print("database groups not mapped to a family (top 10):\n" + unknown.to_string())
    return u.drop(columns=["header", "term", "year"])


# ---------------------------------------------------------------- EMI by category

def category_means(sp: pd.DataFrame, by: str | list[str], value: str, B: int = 2000,
                   seed: int = cfg.SEED) -> pd.DataFrame:
    """Mean of `value` per category (or per combination, e.g. ["country_name", "year"]) with a percentile
    bootstrap interval over MEPs (clusters = person_id; a speech without person_id is its own cluster).
    One row per category; emi_raw is the plain mean EMI."""
    by = [by] if isinstance(by, str) else list(by)
    rng = np.random.default_rng(seed)
    cl = sp.person_id.astype("string").fillna("uid:" + sp.uid.astype("string"))
    rows = []
    for cat, d in sp.assign(cluster=cl).groupby(by):
        s = d.groupby("cluster")[value].agg(["sum", "size"])
        S, N = s["sum"].to_numpy(), s["size"].to_numpy()
        k = len(S)
        if k > 1:
            draws = rng.integers(0, k, size=(B, k))
            lo, hi = np.percentile(S[draws].sum(axis=1) / N[draws].sum(axis=1), [2.5, 97.5])
        else:
            lo = hi = np.nan
        cat = cat if isinstance(cat, tuple) else (cat,)
        rows.append((*cat, len(d), int(d.person_id.nunique()), d[value].mean(), lo, hi, d.emi.mean()))
    return pd.DataFrame(rows, columns=by + ["n_speeches", "n_meps", "mean", "ci_low", "ci_high", "emi_raw"])


def small_multiples(e: pd.DataFrame, ref: pd.DataFrame, label: str, order: list[str], ncols: int,
                    height_mm: float, color: str, series_label: str, ref_color: str, ref_label: str):
    """One panel per category with the yearly mean EMI and its interval; the yearly mean of all speeches as a
    reference line in every panel. Same x and y scales in all panels. Returns the figure."""
    from results.style import DOUBLE, GRID, mm  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MultipleLocator  # noqa: PLC0415

    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(mm(DOUBLE), mm(height_mm)), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12}, squeeze=False)
    # y limits from the intervals, robust to the few very wide ones of small cells
    lo, hi = np.nanpercentile(e.ci_low, 1), np.nanpercentile(e.ci_high, 99)
    pad = 0.05 * (hi - lo)
    years = ref.year.to_numpy()
    for i, ax in enumerate(axes.flat):
        if i >= len(order):
            ax.set_visible(False)
            continue
        name = order[i]
        d = e[e[label].eq(name)].set_index("year").reindex(range(years.min(), years.max() + 1))
        ax.axhline(0, color=MUTED, linewidth=0.4, linestyle=(0, (3, 2)), zorder=1)
        (r,) = ax.plot(ref.year, ref["mean"], color=ref_color, linewidth=0.9, linestyle=(0, (3, 1.5)), zorder=2)
        ax.fill_between(d.index, d.ci_low, d.ci_high, color=color, alpha=0.18, linewidth=0, zorder=3)
        (s,) = ax.plot(d.index, d["mean"], color=color, linewidth=0.9, zorder=4)
        ax.set_title(name, fontsize=6.5, pad=2, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=5.5)
    a0 = axes.flat[0]
    a0.set_ylim(lo - pad, hi + pad)
    a0.set_xlim(years.min() - 0.5, years.max() + 0.5)
    a0.xaxis.set_major_locator(MultipleLocator(10))
    a0.xaxis.set_minor_locator(MultipleLocator(2))
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean EMI", fontsize=6)
    # x tick labels on the lowest visible panel of each column
    for c in range(ncols):
        col = [axes[r, c] for r in range(nrows) if axes[r, c].get_visible()]
        if col:
            col[-1].tick_params(labelbottom=True)
    top = axes[0, -1] if axes[0, -1].get_visible() else axes[0, 0]
    top.legend([s, r], [series_label, ref_label], loc="lower right", bbox_to_anchor=(1, 1.18), ncol=2,
               handlelength=1.8, fontsize=6, borderaxespad=0)
    fig.supxlabel("Year", fontsize=7, y=0.02)
    return fig


# ---------------------------------------------------------------- data for Figures 4 and 5

def latest_scores() -> Path:
    runs = sorted(cfg.SCORES.glob(f"{cfg.MODEL_SHORT}_{cfg.CHUNK_MAX_WORDS}_*/speech_emi.parquet"))
    if not runs:
        raise SystemExit("No EMI scores: run first  uv run python code/scores/score_unified.py")
    return runs[-1].parent


def speeches(scores: str | None, without_human: bool) -> pd.DataFrame:
    """MEP oral speeches with an English text, pooled, with EMI, country and family.
    Prints the checks of the assignment."""
    from results.style import series  # noqa: PLC0415
    from translation.unified import load  # noqa: PLC0415

    folder = cfg.SCORES / scores if scores else latest_scores()
    keys = [k for k, _, _ in series(without_human)]
    u = load("mep-oral")
    sp = pd.read_parquet(folder / "speech_emi.parquet", columns=["uid", "emi"])
    sp = sp.merge(u[["uid", "year", "group"]], on="uid", how="inner")
    sp = sp[sp.group.isin(keys)]
    print(f"EMI from {folder.relative_to(cfg.ROOT)}: {len(sp):,} speeches {sp.group.value_counts().to_dict()}")
    sp = sp.merge(attributes(sp.uid), on="uid", how="left")

    print("\ncountry:", sp.country_check.value_counts().to_dict())
    early = sp[sp.country_check.eq("before accession")]
    if len(early):
        print("speeches before EU accession (left out of the country figures), by country:")
        print(early.groupby("country").year.agg(["size", "min", "max"]).to_string())
    print("\npolitical group from:", sp.family_source.value_counts().to_dict())
    print(pd.crosstab(sp.year, sp.family_source, normalize="index").round(3).to_string())
    return sp


def yearly_reference(sp: pd.DataFrame) -> pd.DataFrame:
    """Mean EMI per year of all speeches of the population (the reference line of Figures 4 and 5)."""
    return sp.groupby("year").emi.mean().rename("mean").reset_index()


def drop_small_cells(e: pd.DataFrame, min_speeches: int, min_meps: int) -> pd.DataFrame:
    """Year-category cells with too few speeches or MEPs get no mean (a gap in the line)."""
    small = (e.n_speeches < min_speeches) | (e.n_meps < min_meps)
    print(f"cells with < {min_speeches} speeches or < {min_meps} MEPs, not drawn: {small.sum():,} of {len(e):,}")
    e = e.copy()
    e.loc[small, ["mean", "ci_low", "ci_high"]] = np.nan
    return e
