# europarl-emi

**Evidence versus intuition in European Parliament speeches, 1996–2026.**

This repository measures whether speeches in the European Parliament rely more on the language of *evidence* (data, research, proof, facts) or of *intuition* (belief, feeling, opinion, common sense), and how this balance has changed over time. It extends to the European Parliament the approach of Aroyehun et al. (2025), who documented a shift from evidence-based to intuition-based language in US congressional speeches.

The repository contains only the code that builds the current data and the paper figures: the download of speeches from the Parliament's API, the unified corpus + API dataset, the machine translation of non-English speeches with eTranslation, and the figure scripts. The earlier pipeline (steps 1–7, review and sensitivity code, legacy notebooks) is in the git history up to commit `ba52117`.

## Data

Data files are **not included** in this repository.

| Path | Content | Produced by |
|---|---|---|
| `data/raw/corpus/all_years_combined.csv.gzip` | European Parliament speeches, all languages, 1996–2023 (621,513 records) | inherited corpus |
| `data/raw/meps/meps_term{4..10}.csv`, `data/processed/mep_database_1994_present.csv` | MEP membership by parliamentary term | [European Parliament Open Data Portal](https://data.europarl.europa.eu/en/datasets) |
| `data/raw/lexicons/*.csv` | Evidence (49) and intuition (35) keyword dictionaries, included in git | Aroyehun et al. (2025) |
| `data/interim/02–03_*.parquet`, `data/processed/06–07_*.parquet` | outputs of the earlier pipeline, read by the API scripts to classify corpus speeches | earlier pipeline (git history) |
| `data/raw/api/speeches/YYYY-MM-DD.jsonl` | speeches from the Parliament's Open Data API, from July 2021 | `download_speeches.py` |
| `data/review/overlap/pairs.parquet` | the same speech matched in corpus and API (July 2021–2023) | `overlap_corpus_api.py` |
| `data/raw/unified/unified.parquet` | one row per speech: corpus to 30-06-2021, API from 01-07-2021 | `build_unified.py` |
| `data/interim/etranslation/unified.sqlite` | eTranslation segments and translations of the non-English texts of `unified`, from 2013 | `translate_unified.py` |

## Code

```
src/europarl_emi/
  config.py                    paths and parameters
  headers.py                   parse_header: role, original language, written statement, group from the speech header
  emi.py                       EMI measure: chunks, MiniLM similarities to the lexicons, length correction, z-score
  etranslation/translate_unified.py   eTranslation client: prepare, run, status, export
notebooks/review/api/
  download_speeches.py         download from https://data.europarl.europa.eu/api/v2/speeches
  overlap_corpus_api.py        match corpus and API speeches on the overlap
  build_unified.py             unified corpus + API dataset
notebooks/paper/
  nhb_style.py                 figure style (Nature Human Behaviour) and project palette
  fig1.py                      Figure 1
scripts/etranslation.sh        runs the eTranslation translation in the background (Mac)
```

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/jacoponudo/europarl-emi.git
cd europarl-emi
uv sync
```

Data, in order:

```bash
uv run python notebooks/review/api/download_speeches.py
uv run python notebooks/review/api/overlap_corpus_api.py
uv run python notebooks/review/api/build_unified.py
uv run --with lingua-language-detector --with tqdm python -m europarl_emi.etranslation.translate_unified prepare \
    --input data/raw/unified/unified.parquet --id-col uid --text-col text --lang-col original_language \
    --date-col date --from-year 2013
sh scripts/etranslation.sh start
```

Figures:

```bash
uv run python notebooks/paper/fig1.py        # -> results/figures/paper/fig1.pdf / .png
```

## Measure

Each speech is split into chunks of up to 150 words, embedded with the sentence-transformer `all-MiniLM-L6-v2`, and compared (cosine similarity) with the mean embedding of each lexicon. Similarities are adjusted for chunk length (10-word bins) and standardised over all chunks analysed together; `EMI = z(evidence) − z(intuition)`, and the speech EMI is the mean over its chunks. Positive values mean more evidence-like language.

## Figures

**Figure 1.** (a) MEP oral speeches per year, delivered in English or translated with eTranslation. (b) Mean EMI per year with percentile bootstrap 95% intervals (B = 2000), for all speeches, speeches delivered in English, and translated speeches.

## References

- Aroyehun, S. T., Simchon, A., Carrella, F., Lasser, J., Lewandowsky, S. & Garcia, D. Computational analysis of US congressional speeches reveals a shift from evidence to intuition. *Nature Human Behaviour* **9**, 1122–1133 (2025). https://doi.org/10.1038/s41562-025-02136-2

## Contact

Jacopo Nudo
