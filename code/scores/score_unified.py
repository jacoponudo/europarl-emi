"""EMI scores of the speeches of the unified dataset, saved once and read by every figure script.

Speeches scored: those with an English text: delivered in English, translated by hand by the Parliament
(to 2012) or translated with eTranslation (from 2013); see code/translation/unified.py.
--population mep-oral (default) or all.
Measure (emi.py, as in the paper baseline): spaces normalised, speeches with more than 10 words, chunks of
<= 150 words (a last chunk < 50 words joins the previous one), header included, MiniLM all-MiniLM-L6-v2,
cosine with the mean embedding of each lexicon; length correction in 10-word bins and z-score computed ONCE
on all chunks scored here; EMI = z(evidence) - z(intuition); speech EMI = mean of its chunks.

    uv run python code/scores/score_unified.py                     # folder minilm_150_<today>
    uv run python code/scores/score_unified.py --date 20261119     # reuse / complete that folder

Out: data/scores/<model>_<window>_<date>/
    speech_emi.parquet   uid, emi, evidence_z, intuition_z, n_chunks, n_words (one row per speech)
    meta.json            measure, population, counts
    chunk_scores/        raw similarities per chunk (cache: the slow part). Chunks already scored are reused,
                         keyed by (uid, chunk_id), and only new ones are embedded; so adding speeches to a
                         folder costs only their embeddings, and EMI is then recomputed on all chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
import config as cfg  # noqa: E402
from scores.emi import compute_emi, split_into_chunks  # noqa: E402
from translation.unified import load  # noqa: E402

PART_SIZE = 20_000
GROUPS = ["original", "human", "etranslation"]  # English texts scored (see translation/unified.py)


def folder(day: str) -> Path:
    return cfg.SCORES / f"{cfg.MODEL_SHORT}_{cfg.CHUNK_MAX_WORDS}_{day}"


def chunk_table(u: pd.DataFrame) -> pd.DataFrame:
    s = u[u.group.isin(GROUPS)].sort_values("uid")
    text = s.en.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    keep = text.str.split().str.len() > cfg.SCORE_MIN_WORDS
    rows = [(uid, i, c) for uid, t in zip(s.uid[keep], text[keep]) for i, c in enumerate(split_into_chunks(t))]
    ch = pd.DataFrame(rows, columns=["uid", "chunk_id", "chunk_text"])
    ch["n_words"] = ch.chunk_text.str.split().str.len()
    print(f"speeches to score: {keep.sum():,} (dropped with <= {cfg.SCORE_MIN_WORDS} words: {(~keep).sum():,}); "
          f"chunks: {len(ch):,}")
    return ch


def score(ch: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Raw similarities per chunk. Chunks already in the cache (same uid, chunk_id and n_words) are reused;
    only the others are embedded, in new parts. Resumes if interrupted."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("model") != cfg.EMBEDDING_MODEL:
        raise SystemExit(f"{cache} was scored with another model: use a new --date.")
    manifest.write_text(json.dumps({"model": cfg.EMBEDDING_MODEL, "part_size": PART_SIZE}))

    parts = sorted(cache.glob("part-*.parquet"))
    key = ["uid", "chunk_id"]
    old = (pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates(key)
           if parts else pd.DataFrame(columns=key + ["n_words", "sim_evidence", "sim_intuition"]))
    m = ch.merge(old, on=key, how="left", suffixes=("", "_cached"))
    bad = m.n_words_cached.notna() & m.n_words_cached.ne(m.n_words)
    if bad.any():
        raise SystemExit(f"{int(bad.sum()):,} cached chunks have a different length (texts changed): use a new --date.")
    todo = ch[m.sim_evidence.isna().to_numpy()]
    print(f"chunks: {len(ch):,}; already scored: {len(ch) - len(todo):,}; to score: {len(todo):,}")
    if len(todo):
        from sentence_transformers import SentenceTransformer, util
        from tqdm import tqdm

        from scores.emi import lexicon_vectors, pick_device

        device = pick_device()
        model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
        ref = lexicon_vectors(model)
        first = 1 + max((int(p.stem.split("-")[1]) for p in parts), default=-1)
        n_new = math.ceil(len(todo) / PART_SIZE)
        print(f"scoring {n_new} new parts on {device}")
        for i in tqdm(range(n_new), desc="parts"):
            part = todo.iloc[i * PART_SIZE:(i + 1) * PART_SIZE].copy()
            emb = model.encode(part.chunk_text.tolist(), batch_size=128, convert_to_tensor=True)
            sims = util.cos_sim(emb, ref).cpu().numpy()
            part["sim_evidence"], part["sim_intuition"] = sims[:, 0], sims[:, 1]
            path = cache / f"part-{first + i:04d}.parquet"
            part.drop(columns="chunk_text").to_parquet(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").rename(path)
        old = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("part-*.parquet"))],
                        ignore_index=True).drop_duplicates(key)
    s = ch.drop(columns="chunk_text").merge(old[key + ["sim_evidence", "sim_intuition"]], on=key, how="left")
    assert s.sim_evidence.notna().all(), "some chunks have no similarity"
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
        "population": args.population,
        "texts": "delivered in English + human translation by the Parliament (to 2012) + eTranslation (from 2013)",
        "groups_scored": GROUPS,
        "standardisation": "length correction (10-word bins) and z-score over all chunks of this folder",
        "n_speeches": len(sp), "n_chunks": len(s), "speeches_by_group": groups,
        "sd_speech_emi": round(float(sp.emi.std()), 4),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"speeches with EMI: {len(sp):,} {groups}; SD of speech EMI: {sp.emi.std():.3f}")
    print(f"saved {(out / 'speech_emi.parquet').relative_to(cfg.ROOT)} and meta.json")


if __name__ == "__main__":
    main()
