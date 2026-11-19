"""Paths and parameters, in one place. The folders of code/ mirror those of data/.

data/
  raw/inherited/   corpus, MEP membership and identifiers, lexicons (as received)
  raw/api/         speeches downloaded from the Parliament's Open Data API
  translation/     unified.parquet (corpus + API) and etranslation.sqlite (translations of its non-English texts)
  scores/          one folder per EMI methodology, <model>_<window>_<date>: speech_emi.parquet (uid, EMI)
results/           everything else: figures, tables, logs
"""

from pathlib import Path

# Folders
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_INHERITED = DATA / "raw" / "inherited"
RAW_CORPUS = RAW_INHERITED / "corpus"
RAW_MEPS = RAW_INHERITED / "meps"
LEXICONS = RAW_INHERITED / "lexicons"
RAW_API = DATA / "raw" / "api" / "speeches"
TRANSLATION = DATA / "translation"
SCORES = DATA / "scores"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
LOGS = RESULTS / "logs"

# Files
CORPUS = RAW_CORPUS / "all_years_combined.csv.gzip"
MEP_DB = RAW_MEPS / "mep_database_1994_present.csv"
UNIFIED = TRANSLATION / "unified.parquet"
ETRANSLATION_DB = TRANSLATION / "etranslation.sqlite"

# Reproducibility
SEED = 42

# EMI measure (code/scores/)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SHORT = "minilm"  # first part of the scores folder name
# Models selectable with score_unified.py --model <short name> (sensitivity of the measure)
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # baseline; reads up to 256 tokens
    "mpnet": "sentence-transformers/all-mpnet-base-v2",  # 384 tokens
    "bge": "BAAI/bge-base-en-v1.5",                      # 512 tokens
    "gte": "thenlper/gte-base",                          # 512 tokens
}
SCORE_MIN_WORDS = 10
CHUNK_MAX_WORDS = 150
CHUNK_MIN_WORDS = 50

# Start of each parliamentary term (constitutive sitting); used by build_unified.py
TERM_STARTS = {
    "4th": "1994-07-19", "5th": "1999-07-20", "6th": "2004-07-20", "7th": "2009-07-14",
    "8th": "2014-07-01", "9th": "2019-07-02", "10th": "2024-07-16",
}
