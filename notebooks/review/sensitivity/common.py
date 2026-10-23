"""REVIEW ONLY. Code shared by the sensitivity variants (v1_mpnet.py, v2_word2vec.py).

It imports functions from the main pipeline but never modifies it: it reads
data/interim/03_speeches_substantive.parquet and writes only to
data/review/sensitivity/<variant>/:
    settings.json                    what was changed with respect to the baseline
    chunk_scores/part-XXXX.parquet   similarities per chunk (resumable: rerun the same command)
    chunk_emi.parquet                EMI per chunk (length correction + global z-score, as in step 5)
    speech_emi.parquet               EMI per speech (mean of its chunks)
"""

import argparse
import json
import math
import time

import pandas as pd
from tqdm import tqdm

from europarl_emi import config as cfg
from europarl_emi.step05_compute_emi import compute_emi
from europarl_emi.step06_parse_headers import HEADER

SPEECHES = cfg.INTERIM / "03_speeches_substantive.parquet"
OUT_ROOT = cfg.ROOT / "data" / "review" / "sensitivity"
PART_SIZE = 20_000
SMOKE_FRACTION = 0.005  # --smoke: 0.5% of speeches per year, to check that everything works and time it


def strip_header(text):
    """Remove 'Name (GROUP), role. - (XX)' from the start of a speech; keep the text if no header is found."""
    m = HEADER.match(text or "")
    return text[m.end():].strip() if m else text


def split_into_chunks(text, size, min_last):
    """Chunks of at most `size` words; a final chunk shorter than `min_last` joins the previous one."""
    words = text.split()
    chunks = [words[i:i + size] for i in range(0, len(words), size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_last:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + last
    return [" ".join(c) for c in chunks]


def make_chunks(df, s):
    texts = df["text"].map(strip_header) if s["strip_header"] else df["text"]
    keep = texts.str.split().str.len() > cfg.SCORE_MIN_WORDS
    rows = [
        (guid, i, chunk)
        for guid, text in zip(df.loc[keep, "guid"], texts[keep])
        for i, chunk in enumerate(split_into_chunks(text, s["chunk_max"], s["chunk_min"]))
    ]
    chunks = pd.DataFrame(rows, columns=["guid", "chunk_id", "chunk_text"])
    chunks["n_words"] = chunks["chunk_text"].str.split().str.len()
    return chunks


def run(name, settings, make_scorer):
    """make_scorer() loads the model and returns a function: list of texts -> (sim_evidence, sim_intuition)."""
    parser = argparse.ArgumentParser(description=f"Sensitivity variant {name} (review only)")
    parser.add_argument("--sample", type=float, default=None, help="fraction of speeches per year, e.g. 0.1")
    parser.add_argument("--smoke", action="store_true", help=f"quick test on {SMOKE_FRACTION:.1%} of speeches")
    args = parser.parse_args()
    fraction = SMOKE_FRACTION if args.smoke else args.sample
    folder = f"_smoke_{name}" if args.smoke else name if fraction is None else f"{name}_sample{round(fraction * 100)}"
    out_dir = OUT_ROOT / folder
    parts = out_dir / "chunk_scores"
    parts.mkdir(parents=True, exist_ok=True)
    for stale in parts.glob("*.tmp"):  # left by an interrupted run
        stale.unlink()
    (out_dir / "settings.json").write_text(json.dumps({"variant": name, "fraction": fraction, **settings}, indent=2))
    print(f"Variant {name}: {settings}")

    speeches = pd.read_parquet(SPEECHES, columns=["guid", "text", "year"])
    if fraction is not None:
        speeches = speeches.groupby("year", group_keys=False).sample(frac=fraction, random_state=cfg.SEED)
        print(f"Sample: {fraction:.1%} of speeches in every year")
    chunks = make_chunks(speeches, settings)
    print(f"Speeches: {chunks['guid'].nunique():,}   chunks: {len(chunks):,}   output: {out_dir.relative_to(cfg.ROOT)}")

    n_parts = math.ceil(len(chunks) / PART_SIZE)
    todo = [p for p in range(n_parts) if not (parts / f"part-{p:04d}.parquet").exists()]
    if todo:
        start = time.time()
        score = make_scorer()
        print(f"Model loaded in {time.time() - start:.0f} s")
        start = time.time()
        for p in tqdm(todo, desc="Parts", mininterval=30):
            part = chunks.iloc[p * PART_SIZE:(p + 1) * PART_SIZE].copy()
            part["sim_evidence"], part["sim_intuition"] = score(part["chunk_text"].tolist())
            path = parts / f"part-{p:04d}.parquet"
            tmp = path.with_suffix(".tmp")
            part.drop(columns="chunk_text").to_parquet(tmp, index=False)
            tmp.rename(path)
        elapsed = time.time() - start
        print(f"Scoring took {elapsed / 60:.1f} min")
        if fraction is not None:
            print(f"Estimated time for the full corpus: about {elapsed / fraction / 3600:.1f} hours")

    scores = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part-*.parquet"))], ignore_index=True)
    assert len(scores) == len(chunks), "some parts are missing: rerun the same command, it resumes"
    empty = scores["sim_evidence"].isna() | scores["sim_intuition"].isna()
    if empty.any():
        print(f"Chunks without a usable representation (dropped): {empty.sum():,} ({empty.mean():.2%})")
    scores = compute_emi(scores[~empty].copy())
    scores.to_parquet(out_dir / "chunk_emi.parquet", index=False)

    speech = scores.groupby("guid").agg(
        emi=("emi", "mean"), evidence_z=("evidence_z", "mean"), intuition_z=("intuition_z", "mean"),
        n_chunks=("chunk_id", "size"), n_words=("n_words", "sum"),
    ).reset_index()
    speech.to_parquet(out_dir / "speech_emi.parquet", index=False)
    print(f"Saved {(out_dir / 'speech_emi.parquet').relative_to(cfg.ROOT)} ({len(speech):,} speeches)")
