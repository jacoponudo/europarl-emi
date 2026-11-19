"""One-off: move data/ to the new layout and delete what is no longer needed.

New layout
  data/raw/inherited/   corpus/, meps/ (membership + mep_database_1994_present.csv with the MEP ids),
                        lexicons/, legacy_colab/ (outputs of the original Colab code, not regenerable)
  (data/review/etranslation, translation checks, goes to results/review/etranslation)
  data/raw/api/         untouched
  data/translation/     unified.parquet, etranslation.sqlite (+ etranslation_export.parquet if present)
  data/scores/<model>_<window>_<date>/   only scores computed on unified (score_unified.py): the chunk cache of the
      fig1 run is moved here. Scores computed on the old corpus (pipeline baseline 05_speech_emi.parquet and the
      sensitivity variants in data/review/sensitivity/) are deleted.
  results/              tables, logs and archive (data/archive) moved out of data/
Everything else in data/ (old pipeline outputs 01-07, old scores, review, analysis, overlap, old eTranslation v1)
is deleted.

    uv run python scripts/reorganize_data.py            # dry run: prints the plan, changes nothing
    uv run python scripts/reorganize_data.py --apply    # does it
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
R = ROOT / "results"
KEEP = [D / "raw" / "inherited", D / "raw" / "api", D / "translation", D / "scores"]

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


# ---------------------------------------------------------------- plan

def build_plan() -> Plan:
    pl = Plan()
    inh = D / "raw" / "inherited"
    for f in ("data_english_simplified.csv.gzip", "dataenglishv2.csv.gzip"):  # English subsets made by the Colab code
        pl.move(D / "raw" / "corpus" / f, inh / "legacy_colab" / f)
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
    pl.move_children(D / "review" / "etranslation", R / "review" / "etranslation")

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
    return subprocess.run(["pgrep", "-f", "translate_unified run|translation.etranslation run"], capture_output=True).returncode == 0


def checkpoint(db: Path) -> None:
    """Write the -wal journal into the database file, so that the .sqlite alone holds everything."""
    import sqlite3

    if not db.exists():
        return
    con = sqlite3.connect(db)
    busy, log, done = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    n = con.execute("SELECT COUNT(*) FROM segments WHERE status = 'done'").fetchone()[0]
    con.close()
    if busy:
        raise SystemExit(f"{rel(db)} is in use: close every program using it and run again.")
    print(f"{rel(db)}: journal written into the database ({done} pages); translated segments: {n:,}")


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
    print("\nDELETE (not in the new layout)")
    if dels:
        t = summarise_deletions(dels)
        for r in t.itertuples():
            print(f"  {r.folder:<50} {r.files:>7,} files  {human(r.size):>9}")
            if r.files <= 12:  # few files: show them
                for f in dels:
                    parts = f.relative_to(D).parts
                    g = "data/" + "/".join(parts[:2] if len(parts) > 2 else parts[:1])
                    if g == r.folder:
                        print(f"      {f.relative_to(D / r.folder.removeprefix('data/'))}  {human(f.stat().st_size)}")
        print(f"  {'total':<50} {len(dels):>7,} files  {human(t['size'].sum()):>9}")

    if not args.apply:
        print("\nDry run: nothing changed. Check the lists, then run again with --apply.")
        return
    if etranslation_busy():
        raise SystemExit("eTranslation seems to be running: stop it first (sh scripts/etranslation.sh stop).")
    checkpoint(D / "interim" / "etranslation" / "unified.sqlite")

    for s, d in pl.moves:
        if d.exists():
            print(f"skipped (destination exists): {rel(s)}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    for f in dels:
        f.unlink(missing_ok=True)
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
