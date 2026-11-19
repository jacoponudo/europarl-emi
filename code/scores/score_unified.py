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
    uv run python code/scores/score_unified.py --model bge --window 300 --max-seq-length 512 --date 20261119
                                                                   # another methodology: folder bge_300_20261119
--model is a short name of config.MODELS (default minilm), --window the words per chunk (default 150; a last
chunk < 50 words still joins the previous one). --max-seq-length raises the model's token limit (default: the
model's own, e.g. 256 for MiniLM): with 300-word chunks (~400 tokens) use 512, or the chunk end is cut off.
The share of chunks longer than the token limit is measured on a sample and written to meta.json.

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


def truncation(model, ch: pd.DataFrame, n: int = 5_000) -> dict:
    """Share of chunks longer than the model's token limit (their end is not read), on a random sample."""
    sample = ch.chunk_text.sample(min(n, len(ch)), random_state=cfg.SEED).tolist()
    lengths = pd.Series([len(ids) for ids in model.tokenizer(sample, add_special_tokens=True)["input_ids"]])
    limit = int(model.max_seq_length)
    return {"max_seq_length": limit, "tokens_per_chunk_median": int(lengths.median()),
            "tokens_per_chunk_p95": int(lengths.quantile(0.95)),
            "share_chunks_truncated": round(float(lengths.gt(limit).mean()), 4), "truncation_sample": len(sample)}


def score(ch: pd.DataFrame, cache: Path, max_seq_length: int | None = None,
          batch_size: int = 128) -> tuple[pd.DataFrame, dict]:
    """Raw similarities per chunk. Chunks already in the cache (same uid, chunk_id and n_words) are reused;
    only the others are embedded, in new parts. Resumes if interrupted."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    info = json.loads(manifest.read_text()) if manifest.exists() else {}
    if info and info.get("model") != cfg.EMBEDDING_MODEL:
        raise SystemExit(f"{cache} was scored with another model: use a new --date.")
    if info and info.get("max_seq_length_arg") != max_seq_length:
        raise SystemExit(f"{cache} was scored with --max-seq-length {info.get('max_seq_length_arg')}: "
                         "use the same value or a new --date.")
    info.update({"model": cfg.EMBEDDING_MODEL, "part_size": PART_SIZE, "max_seq_length_arg": max_seq_length})
    manifest.write_text(json.dumps(info))

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
        import torch

        # float32 everywhere: some checkpoints (gte-base) are stored in float16, slow and less precise on CPU
        model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device, model_kwargs={"dtype": torch.float32})
        if max_seq_length:
            model.max_seq_length = max_seq_length
        info.update(truncation(model, ch))
        manifest.write_text(json.dumps(info))
        print(f"token limit {info['max_seq_length']}; tokens per chunk: median {info['tokens_per_chunk_median']}, "
              f"95th pct {info['tokens_per_chunk_p95']}; chunks truncated: {info['share_chunks_truncated']:.1%}")
        ref = lexicon_vectors(model)
        first = 1 + max((int(p.stem.split("-")[1]) for p in parts), default=-1)
        n_new = math.ceil(len(todo) / PART_SIZE)
        print(f"scoring {n_new} new parts on {device}")
        for i in tqdm(range(n_new), desc="parts"):
            part = todo.iloc[i * PART_SIZE:(i + 1) * PART_SIZE].copy()
            emb = model.encode(part.chunk_text.tolist(), batch_size=batch_size, convert_to_tensor=True)
            sims = util.cos_sim(emb, ref).cpu().numpy()
            part["sim_evidence"], part["sim_intuition"] = sims[:, 0], sims[:, 1]
            path = cache / f"part-{first + i:04d}.parquet"
            part.drop(columns="chunk_text").to_parquet(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").rename(path)
        old = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("part-*.parquet"))],
                        ignore_index=True).drop_duplicates(key)
    s = ch.drop(columns="chunk_text").merge(old[key + ["sim_evidence", "sim_intuition"]], on=key, how="left")
    assert s.sim_evidence.notna().all(), "some chunks have no similarity"
    return s, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", choices=["mep-oral", "all"], default="mep-oral")
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="date in the folder name (YYYYMMDD)")
    ap.add_argument("--model", choices=sorted(cfg.MODELS), default=cfg.MODEL_SHORT, help="short name, see config.MODELS")
    ap.add_argument("--window", type=int, default=cfg.CHUNK_MAX_WORDS, help="maximum words per chunk")
    ap.add_argument("--max-seq-length", type=int, default=None, help="token limit of the model (default: its own)")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    # the measure reads these from config (emi.split_into_chunks, folder name, manifest)
    cfg.MODEL_SHORT, cfg.EMBEDDING_MODEL, cfg.CHUNK_MAX_WORDS = args.model, cfg.MODELS[args.model], args.window
    print(f"model {cfg.EMBEDDING_MODEL}, window {cfg.CHUNK_MAX_WORDS} words, folder {folder(args.date).name}")

    out = folder(args.date)
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("population") != args.population:
        raise SystemExit(f"{out} was scored with another population: use a new --date.")
    u = load(args.population, with_text=True)
    ch = chunk_table(u)
    s, info = score(ch, out / "chunk_scores", args.max_seq_length, args.batch_size)
    s = compute_emi(s)
    sp = s.groupby("uid").agg(emi=("emi", "mean"), evidence_z=("evidence_z", "mean"),
                              intuition_z=("intuition_z", "mean"), n_chunks=("chunk_id", "size"),
                              n_words=("n_words", "sum")).reset_index()
    sp.to_parquet(out / "speech_emi.parquet", index=False)
    groups = u.set_index("uid").group.reindex(sp.uid).value_counts().to_dict()
    meta = {
        "folder": out.name, "date": args.date, "model": cfg.EMBEDDING_MODEL,
        "window_words": cfg.CHUNK_MAX_WORDS, "last_chunk_min_words": cfg.CHUNK_MIN_WORDS,
        "speech_min_words": cfg.SCORE_MIN_WORDS + 1, "header": "included",
        **{k: info[k] for k in ("max_seq_length", "tokens_per_chunk_median", "tokens_per_chunk_p95",
                                "share_chunks_truncated", "truncation_sample") if k in info},
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
