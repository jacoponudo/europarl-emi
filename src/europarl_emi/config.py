"""Paths and parameters, in one place."""

from pathlib import Path

# Folders
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_CORPUS = DATA / "raw" / "corpus"
RAW_MEPS = DATA / "raw" / "meps"
LEXICONS = DATA / "raw" / "lexicons"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "results" / "figures"

# Reproducibility
SEED = 42

# EMI measure (src/europarl_emi/emi.py)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCORE_MIN_WORDS = 10
CHUNK_MAX_WORDS = 150
CHUNK_MIN_WORDS = 50

# Start of each parliamentary term (constitutive sitting); used by the API scripts
TERM_STARTS = {
    "4th": "1994-07-19", "5th": "1999-07-20", "6th": "2004-07-20", "7th": "2009-07-14",
    "8th": "2014-07-01", "9th": "2019-07-02", "10th": "2024-07-16",
}
