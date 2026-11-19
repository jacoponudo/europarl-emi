"""One-off: move data/ to the new layout and delete what is no longer needed.

New layout
  data/raw/inherited/   corpus/, meps/ (membership + mep_database_1994_present.csv with the MEP ids),
                        lexicons/, legacy_colab/ (outputs of the original Colab code, not regenerable)
  data/raw/api/         untouched
  data/translation/     unified.parquet, etranslation.sqlite (+ etranslation_export.parquet if present)
  data/scores/<model>_<window>_<date>/   speech_emi.parquet (uid, emi, ...) + meta.json, one folder per methodology:
      the new scores of score_unified.py (chunk cache moved from the fig1 run), the old pipeline baseline
      (data/processed/05_speech_emi.parquet) and the sensitivity variants (data/review/sensitivity/<variant>/)
  results/              tables, logs and archive (data/archive) moved out of data/
Everything else in data/ (old pipeline outputs 01-07, review, analysis, overlap, old eTranslation v1) is deleted.

Ids: uid as in unified ("corpus:<guid>" or "api:<speech_id>"); old files keyed by guid get "corpus:" + guid.

    uv run python scripts/reorganize_data.py            # dry run: prints the plan, changes nothing
    uv run python scripts/reorganize_data.py --apply    # does it
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
R = ROOT / "results"
KEEP = [D / "raw" / "inherited", D / "raw" / "api", D / "translation", D / "scores"]

# sensitivity variant -> methodology name (model_window) and what differs from the paper baseline
VARIANTS = {
    "v1_mpnet": ("mpnet_50", "all-mpnet-base-v2, 50-word windows, header removed"),
    "v2_word2vec": ("word2vec_50", "word2vec Google News (mean of vectors), 50-word windows, header removed"),
    "v2b_word2vec_p80": ("word2vec-p80_50", "word2vec, 80th percentile of word similarities, 50-word windows"),
    "v3_mpnet_only": ("mpnet_150", "all-mpnet-base-v2, otherwise as the baseline"),
    "v4_lexicon_count": ("lexcount_100", "no model: lexicon words per 100 words"),
    "v5_minilm_noheader": ("minilm-noheader_150", "as the baseline, header removed"),
}


def size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def day(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


class Plan:
    def __init__(self):
        self.moves: list[tuple[Path, Path]] = []
        self.extracts: list[tuple[Path, Path, dict]] = []
        self.handled: set[Path] = set()

    def move(self, src: Path, dst: Path):
        if src.exists() and src not in self.handled:
            self.moves.append((src, dst))
            self.handled.add(src)

    def move_children(self, src_dir: Path, dst_dir: Path, pattern: str = "*"):
        if src_dir.is_dir():
            for f in sorted(src_dir.glob(pattern)):
                if f.name != ".gitkeep":
                    self.move(f, dst_dir / f.name)


# ---------------------------------------------------------------- scores found in old files

def read_cols(p: Path) -> list[str]:
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_schema(p).names
    return list(pd.read_csv(p, nrows=0).columns)


def read(p: Path, cols: list[str]) -> pd.DataFrame:
    return pd.read_parquet(p, columns=cols) if p.suffix == ".parquet" else pd.read_csv(p, usecols=cols)


def find_scores(folder: Path):
    """A file with an id column (uid/guid) and an EMI column; speech level preferred, else chunk level."""
    best = None
    files = [f for f in folder.rglob("*") if f.is_file() and (f.suffix == ".parquet" or ".csv" in f.name)]
    for f in sorted(files):
        try:
            cols = read_cols(f)
        except Exception:
            continue
        idc = next((c for c in ("uid", "guid") if c in cols), None)
        emic = "emi" if "emi" in cols else next((c for c in cols if c.lower().startswith("emi")), None)
        if not idc or not emic:
            continue
        ids = read(f, [idc])[idc]
        level = "speech" if ids.is_unique else "chunk"
        cand = (level == "speech", -len(ids), f, idc, emic, level, len(ids), ids.nunique())
        if best is None or cand[:2] > best[:2]:
            best = cand
    return None if best is None else dict(zip(["file", "id", "emi", "level", "rows", "speeches"], best[2:]))


def to_speech_emi(info: dict) -> pd.DataFrame:
    f, idc, emic = info["file"], info["id"], info["emi"]
    cols = [c for c in read_cols(f) if c in {idc, emic, "evidence_z", "intuition_z", "n_chunks", "n_words"}]
    df = read(f, cols).rename(columns={emic: "emi"})
    if info["level"] == "chunk":
        agg = {"emi": ("emi", "mean"), "n_chunks": ("emi", "size")}
        agg.update({c: (c, "mean") for c in ("evidence_z", "intuition_z") if c in df})
        if "n_words" in df:
            agg["n_words"] = ("n_words", "sum")
        df = df.groupby(idc).agg(**agg).reset_index()
    if idc == "guid":
        df.insert(0, "uid", "corpus:" + df.pop("guid").astype(str))
    return df


# ---------------------------------------------------------------- plan

def build_plan() -> Plan:
    pl = Plan()
    inh = D / "raw" / "inherited"
    pl.move_children(D / "raw" / "corpus", inh / "corpus")
    pl.move_children(D / "raw" / "meps", inh / "meps")
    pl.move(D / "processed" / "mep_database_1994_present.csv", inh / "meps" / "mep_database_1994_present.csv")
    if not (inh / "lexicons").exists():
        pl.move_children(D / "raw" / "lexicons", inh / "lexicons")
    for f in ("interim/outputv2.csv.gzip", "processed/outputv2_filled.csv.gzip",
              "processed/political_groups_simplified.csv"):
        pl.move(D / f, inh / "legacy_colab" / Path(f).name)

    pl.move_children(D / "archive", R / "archive")
    pl.move_children(D / "review" / "sensitivity" / "logs", R / "logs" / "sensitivity")

    # translation
    uni = D / "raw" / "unified"
    pl.move(uni / "unified.parquet", D / "translation" / "unified.parquet")
    pl.move_children(uni, R / "tables" / "unified", "*.csv")
    pl.move_children(uni, R / "logs", "*.log")
    etr = D / "interim" / "etranslation"
    pl.move(etr / "unified.sqlite", D / "translation" / "etranslation.sqlite")
    pl.move(etr / "unified_translations.parquet", D / "translation" / "etranslation_export.parquet")
    pl.move_children(etr / "logs", R / "logs" / "etranslation")
    for pat in ("*.log", "*.jsonl"):
        pl.move_children(etr, R / "logs" / "etranslation", pat)

    # new scores: chunk cache of the fig1 run
    for old in (D / "paper" / "fig1" / "chunk_scores", D / "analysis" / "02_speeches_by_year_language" / "chunk_scores"):
        man = old / "manifest.json"
        if man.exists():
            pl.move(old, D / "scores" / f"minilm_150_{day(man)}" / "chunk_scores")
            break

    # old pipeline baseline
    base = D / "processed" / "05_speech_emi.parquet"
    if base.exists():
        info = {"file": base, "id": "guid", "emi": "emi", "level": "speech"}
        meta = {"source": rel(base), "model": "sentence-transformers/all-MiniLM-L6-v2", "window_words": 150,
                "header": "included", "population": "English texts of the inherited corpus 1996-2023 after the "
                "procedural filter (paper baseline, old pipeline steps 1-5)",
                "standardisation": "over all chunks of those texts"}
        pl.extracts.append((base, D / "scores" / f"minilm_150_{day(base)}-pipeline", {"info": info, "meta": meta}))
        pl.handled.add(base)

    # sensitivity variants
    sens = D / "review" / "sensitivity"
    if sens.is_dir():
        for v in sorted(p for p in sens.iterdir() if p.is_dir() and p.name != "logs"):
            name, what = VARIANTS.get(v.name, (re.sub(r"^v\d+b?_", "", v.name), "see the variant code (git history)"))
            info = find_scores(v)
            dst = D / "scores" / f"{name}_{day(v)}"
            if info:
                meta = {"source": rel(info["file"]), "variant": v.name, "differs_from_baseline": what,
                        "level_in_source": info["level"]}
                pl.extracts.append((v, dst, {"info": info, "meta": meta}))
                pl.handled.add(v)
            else:  # unknown format: keep the files as they are
                pl.move(v, dst / "original_files")
    return pl


def deletions(pl: Plan) -> list[Path]:
    out = []
    for f in sorted(D.rglob("*")):
        if not f.is_file() or f.name == ".gitkeep":
            continue
        if any(f == h or h in f.parents for h in pl.handled):
            continue
        if any(k in f.parents for k in KEEP):
            continue
        out.append(f)
    return out


def summarise_deletions(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        parts = f.relative_to(D).parts
        group = "data/" + "/".join(parts[:2] if len(parts) > 2 else parts[:1])
        rows.append((group, f.stat().st_size))
    df = pd.DataFrame(rows, columns=["folder", "bytes"])
    return df.groupby("folder").agg(files=("bytes", "size"), size=("bytes", "sum")).reset_index()


def etranslation_busy() -> bool:
    wal = D / "interim" / "etranslation" / "unified.sqlite-wal"
    running = subprocess.run(["pgrep", "-f", "translate_unified run"], capture_output=True).returncode == 0
    return running or (wal.exists() and wal.stat().st_size > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="do it (without: dry run)")
    args = ap.parse_args()

    pl = build_plan()
    dels = deletions(pl)

    print("MOVE")
    for s, d in pl.moves:
        flag = "   (destination exists: SKIPPED)" if d.exists() else ""
        print(f"  {rel(s):<70} -> {rel(d)}  [{human(size(s))}]{flag}")
    print("\nSCORES (id + EMI per speech, written to data/scores/<model>_<window>_<date>/speech_emi.parquet)")
    for s, d, x in pl.extracts:
        i = x["info"]
        print(f"  {rel(i['file'])}  [{i['level']} level, id {i['id']}, EMI column {i['emi']}]  -> {rel(d)}")
    print("\nDELETE (not in the new layout)")
    if dels:
        t = summarise_deletions(dels)
        for r in t.itertuples():
            print(f"  {r.folder:<50} {r.files:>7,} files  {human(r.size):>9}")
        print(f"  {'total':<50} {len(dels):>7,} files  {human(t['size'].sum()):>9}")
    for s, _, _ in pl.extracts:
        print(f"  {rel(s)}  (after extracting the scores)  {human(size(s))}")

    if not args.apply:
        print("\nDry run: nothing changed. Check the lists, then run again with --apply.")
        return
    if etranslation_busy():
        raise SystemExit("eTranslation seems to be running (or the database has unsaved changes): stop it first.")

    for s, d, x in pl.extracts:
        df = to_speech_emi(x["info"])
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "speech_emi.parquet", index=False)
        meta = {**x["meta"], "folder": d.name, "n_speeches": len(df), "sd_speech_emi": round(float(df.emi.std()), 4)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"written {rel(d / 'speech_emi.parquet')}: {len(df):,} speeches")
    for s, d in pl.moves:
        if d.exists():
            print(f"skipped (destination exists): {rel(s)}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    for f in dels:
        f.unlink()
    for s, _, _ in pl.extracts:
        shutil.rmtree(s) if s.is_dir() else s.unlink()
    for p in sorted((p for p in D.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(p.iterdir()) and not any(k == p or k in p.parents or p in k.parents for k in KEEP):
            p.rmdir()
    print("\ndone. data/ now:")
    for top in (D / "raw" / "inherited", D / "raw" / "api", D / "translation", D / "scores"):
        for p in sorted(top.glob("*")):
            if p.name != ".gitkeep":
                print(f"  {rel(p):<55} {human(size(p)):>9}")


if __name__ == "__main__":
    main()
