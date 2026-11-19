"""EMI scores of the speeches of the unified dataset, saved once and read by every figure script.

Speeches scored: those with an English text, delivered in English or translated with eTranslation
(see translation/unified.py); --population mep-oral (default) or all.
Measure (emi.py, as in the paper baseline): spaces normalised, speeches with more than 10 words, chunks of
<= 150 words (a last chunk < 50 words joins the previous one), header included, MiniLM all-MiniLM-L6-v2,
cosine with the mean embedding of each lexicon; length correction in 10-word bins and z-score computed ONCE
on all chunks scored here; EMI = z(evidence) - z(intuition); speech EMI = mean of its chunks.

    uv run python -m europarl_emi.scores.score_unified                     # folder minilm_150_<today>
    uv run python -m europarl_emi.scores.score_unified --date 20261119     # reuse / complete that folder

Out: data/scores/<model>_<window>_<date>/
    speech_emi.parquet   uid, emi, evidence_z, intuition_z, n_chunks, n_words (one row per speech)
    meta.json            measure, population, counts
    chunk_scores/        raw similarities per chunk (cache: the slow part, resumes from the missing parts;
                         if the speeches change, a new date gives a new folder)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from europarl_emi import config as cfg
from europarl_emi.scores.emi import compute_emi, split_into_chunks
from europarl_emi.translation.unified import load

PART_SIZE = 20_000


def folder(day: str) -> Path:
    return cfg.SCORES / f"{cfg.MODEL_SHORT}_{cfg.CHUNK_MAX_WORDS}_{day}"


def chunk_table(u: pd.DataFrame) -> pd.DataFrame:
    s = u[u.group.isin(["original", "etranslation"])].sort_values("uid")
    text = s.en.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    keep = text.str.split().str.len() > cfg.SCORE_MIN_WORDS
    rows = [(uid, i, c) for uid, t in zip(s.uid[keep], text[keep]) for i, c in enumerate(split_into_chunks(t))]
    ch = pd.DataFrame(rows, columns=["uid", "chunk_id", "chunk_text"])
    ch["n_words"] = ch.chunk_text.str.split().str.len()
    print(f"speeches to score: {keep.sum():,} (dropped with <= {cfg.SCORE_MIN_WORDS} words: {(~keep).sum():,}); "
          f"chunks: {len(ch):,}")
    return ch


def score(ch: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Raw similarities per chunk, saved in parts; resumes from the missing parts."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    sig = {"n_chunks": len(ch), "n_speeches": int(ch.uid.nunique()), "first": ch.uid.iloc[0],
           "last": ch.uid.iloc[-1], "part_size": PART_SIZE, "model": cfg.EMBEDDING_MODEL}
    if manifest.exists() and json.loads(manifest.read_text()) != sig:
        raise SystemExit(f"The speeches differ from those scored in {cache}: use a new --date.")
    manifest.write_text(json.dumps(sig))

    n_parts = math.ceil(len(ch) / PART_SIZE)
    todo = [p for p in range(n_parts) if not (cache / f"part-{p:04d}.parquet").exists()]
    if todo:
        from sentence_transformers import SentenceTransformer, util
        from tqdm import tqdm

        from europarl_emi.scores.emi import lexicon_vectors, pick_device

        device = pick_device()
        print(f"scoring {len(todo)} of {n_parts} parts on {device}")
        model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
        ref = lexicon_vectors(model)
        for p in tqdm(todo, desc="parts"):
            part = ch.iloc[p * PART_SIZE:(p + 1) * PART_SIZE].copy()
            emb = model.encode(part.chunk_text.tolist(), batch_size=128, convert_to_tensor=True)
            sims = util.cos_sim(emb, ref).cpu().numpy()
            part["sim_evidence"], part["sim_intuition"] = sims[:, 0], sims[:, 1]
            path = cache / f"part-{p:04d}.parquet"
            part.drop(columns="chunk_text").to_parquet(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").rename(path)
    else:
        print(f"all {n_parts} parts already scored in {cache}")
    s = pd.concat([pd.read_parquet(cache / f"part-{p:04d}.parquet") for p in range(n_parts)], ignore_index=True)
    assert len(s) == len(ch), f"chunks in cache {len(s):,} != expected {len(ch):,}"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", choices=["mep-oral", "all"], default="mep-oral")
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="date in the folder name (YYYYMMDD)")
    args = ap.parse_args()

    out = folder(args.date)
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("population") != args.population:
        raise SystemExit(f"{out} was scored with another population: use a new --date.")
    u = load(args.population, with_text=True)
    ch = chunk_table(u)
    s = compute_emi(score(ch, out / "chunk_scores"))
    sp = s.groupby("uid").agg(emi=("emi", "mean"), evidence_z=("evidence_z", "mean"),
                              intuition_z=("intuition_z", "mean"), n_chunks=("chunk_id", "size"),
                              n_words=("n_words", "sum")).reset_index()
    sp.to_parquet(out / "speech_emi.parquet", index=False)
    groups = u.set_index("uid").group.reindex(sp.uid).value_counts().to_dict()
    meta = {
        "folder": out.name, "date": args.date, "model": cfg.EMBEDDING_MODEL,
        "window_words": cfg.CHUNK_MAX_WORDS, "last_chunk_min_words": cfg.CHUNK_MIN_WORDS,
        "speech_min_words": cfg.SCORE_MIN_WORDS + 1, "header": "included",
        "population": args.population, "texts": "delivered in English + translated with eTranslation",
        "standardisation": "length correction (10-word bins) and z-score over all chunks of this folder",
        "n_speeches": len(sp), "n_chunks": len(s), "speeches_by_group": groups,
        "sd_speech_emi": round(float(sp.emi.std()), 4),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"speeches with EMI: {len(sp):,} {groups}; SD of speech EMI: {sp.emi.std():.3f}")
    print(f"saved {(out / 'speech_emi.parquet').relative_to(cfg.ROOT)} and meta.json")


if __name__ == "__main__":
    main()
