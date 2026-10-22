"""Inventario dei file e primi controlli sui dati."""

import pandas as pd

from europarl_emi import config as C

FILES = {
    "raw/corpus/all_years_combined": C.RAW_CORPUS / "all_years_combined.csv.gzip",
    "raw/corpus/data_english_simplified": C.RAW_CORPUS / "data_english_simplified.csv.gzip",
    "raw/corpus/dataenglishv2": C.RAW_CORPUS / "dataenglishv2.csv.gzip",
    "interim/dataenglishv3": C.INTERIM / "dataenglishv3.csv.gzip",
    "interim/train": C.INTERIM / "train.csv",
    "interim/val": C.INTERIM / "val.csv",
    "interim/filtered_nov2": C.INTERIM / "filtered_nov2.csv",
    "interim/outputv2": C.INTERIM / "outputv2.csv.gzip",
    "processed/mep_database": C.PROCESSED / "mep_database_1994_present.csv",
    "processed/outputv2_filled": C.PROCESSED / "outputv2_filled.csv.gzip",
    "processed/political_groups_simplified": C.PROCESSED / "political_groups_simplified.csv",
}
EMI = "evidence_minus_intuition_score"


def read(path, **kw):
    comp = "gzip" if path.suffix in (".gzip", ".gz") else None
    return pd.read_csv(path, compression=comp, low_memory=False, **kw)


def columns(path):
    return read(path, nrows=0).columns.tolist()


def dates(s):
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


def title(text):
    print("=" * 70)
    print(text)
    print()


def inventory():
    title("1. INVENTARIO")
    for name, path in FILES.items():
        if not path.exists():
            print(f"MANCA  {name}\n")
            continue
        cols = columns(path)
        use = [cols[0]] + [c for c in ("language", "pubdate") if c in cols and c != cols[0]]
        df = read(path, usecols=use)
        size = path.stat().st_size / 1e6
        print(f"{name}  |  {len(df):,} righe  |  {len(cols)} colonne  |  {size:,.0f} MB")
        print("   colonne:", ", ".join(cols))
        if "language" in df:
            print("   lingue:", df["language"].value_counts().head(8).to_dict())
        if "pubdate" in df:
            d = dates(df["pubdate"])
            if d.notna().any():
                print(f"   date: {d.min():%Y-%m-%d} -> {d.max():%Y-%m-%d}  (non leggibili: {d.isna().sum():,})")
        print(flush=True)


def batch_check():
    title("2. ORDINAMENTO E BLOCCHI DA 1000 RIGHE (interim/outputv2)")
    path = FILES["interim/outputv2"]
    cols = columns(path)
    missing = [c for c in ("pubdate", EMI) if c not in cols]
    if missing:
        print("   colonne mancanti:", missing)
        return
    df = read(path, usecols=["pubdate", EMI])
    d = dates(df["pubdate"])
    ordered = (d.diff().dt.total_seconds().dropna() >= 0).mean()
    print(f"   righe consecutive in ordine cronologico: {ordered:.1%}")
    b = pd.DataFrame({"batch": df.index // 1000, "year": d.dt.year, "emi": df[EMI]}).groupby("batch")
    span = b["year"].max() - b["year"].min()
    means = b["emi"].mean()
    print(f"   blocchi: {b.ngroups:,}")
    print(f"   blocchi che coprono un solo anno: {(span == 0).mean():.1%}")
    print(f"   media EMI per blocco: da {means.min():.4f} a {means.max():.4f}")
    yearly = df.groupby(d.dt.year)[EMI].agg(["mean", "std", "count"]).round(3)
    print("\n   EMI per anno:")
    print(yearly.to_string())
    print(flush=True)


def metadata_check():
    title("3. METADATI PARLAMENTARI")
    mep = read(FILES["processed/mep_database"])
    print(f"   database MEP: {len(mep):,} righe, {mep['name'].nunique():,} nomi unici")
    print(f"   colonne: {mep.columns.tolist()}")
    print(f"   start_date compilata: {mep['start_date'].notna().mean():.1%}")
    groups = mep.groupby("name")["political_group"].nunique()
    print(f"   nomi con piu di un gruppo politico: {(groups > 1).sum():,}")
    path = FILES["processed/outputv2_filled"]
    wanted = ("detectedspeaker", "speaker-country", "detectedPoliticalGroupEU")
    use = [c for c in wanted if c in columns(path)]
    df = read(path, usecols=use)
    for c in use:
        print(f"   {c}: mancante nel {df[c].isna().mean():.1%} delle righe")
    print(flush=True)


def lexicon_check():
    title("4. LESSICI")
    for name in ("evidence", "intuition"):
        lex = pd.read_csv(C.LEXICONS / f"{name}_lexicon.csv")
        terms = lex.iloc[:, 0].dropna().astype(str).tolist()
        print(f"   {name}: {len(terms)} voci, colonne {lex.columns.tolist()}")
        print("   " + "; ".join(terms))
        print()


if __name__ == "__main__":
    for step in (inventory, batch_check, metadata_check, lexicon_check):
        try:
            step()
        except Exception as e:
            print(f"   ERRORE in {step.__name__}: {type(e).__name__}: {e}\n")
