# europarl-emi

**Evidence versus intuition in European Parliament speeches, 1996–2026.**

This repository measures whether speeches in the European Parliament rely more on the language of *evidence* (data, research, proof, facts) or of *intuition* (belief, feeling, opinion, common sense), and how this balance has changed over time. It extends to the European Parliament the approach of Aroyehun et al. (2025), who documented a shift from evidence-based to intuition-based language in US congressional speeches.

The repository contains only the code that builds the current data and the paper figures: the download of speeches from the Parliament's API, the unified corpus + API dataset, the machine translation of non-English speeches with eTranslation, the EMI scores and the figure scripts. The earlier pipeline (steps 1–7, review and sensitivity code, legacy notebooks, corpus–API overlap check) is in the git history up to commit `ba52117`.

## Data

Data files are **not included** in this repository (only the lexicons).

```
data/
  raw/inherited/corpus/       all_years_combined.csv.gzip: speeches, all languages, 1996–2023 (621,513 records)
  raw/inherited/meps/         MEP membership by term and mep_database_1994_present.csv (MEP ids)
  raw/inherited/lexicons/     evidence (49) and intuition (35) keywords, Aroyehun et al. (2025), in git
  raw/inherited/legacy_colab/ outputs of the original Colab notebooks (not regenerable)
  raw/api/speeches/           YYYY-MM-DD.jsonl from the Parliament's Open Data API, from July 2021
  translation/                unified.parquet (one row per speech: corpus to 30-06-2021, API from 01-07-2021)
                              etranslation.sqlite (eTranslation of the non-English texts of unified, from 2013)
  scores/<model>_<window>_<date>/   one folder per EMI methodology: speech_emi.parquet (uid, emi, ...) + meta.json
results/                      figures, tables, logs, archive
```

Speech ids are `uid` as in `unified.parquet`: `corpus:<guid>` or `api:<speech_id>`.

## Code

```
src/europarl_emi/
  config.py                    paths and parameters
  headers.py                   parse_header: role, original language, written statement, group from the speech header
  unified.py                   unified dataset + English text of each speech (delivered in English or eTranslation)
  emi.py                       EMI measure: chunks, MiniLM similarities to the lexicons, length correction, z-score
  score_unified.py             EMI scores -> data/scores/minilm_150_<date>/
  etranslation/translate_unified.py   eTranslation client: prepare, run, status, export
notebooks/api/
  download_speeches.py         download from https://data.europarl.europa.eu/api/v2/speeches
  build_unified.py             unified corpus + API dataset
notebooks/paper/
  nhb_style.py                 figure style (Nature Human Behaviour) and project palette
  fig1.py                      Figure 1
scripts/
  etranslation.sh              runs the eTranslation translation in the background (Mac)
  reorganize_data.py           one-off move of data/ to this layout (dry run by default)
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
uv run python notebooks/api/download_speeches.py
uv run python notebooks/api/build_unified.py
uv run --with lingua-language-detector --with tqdm python -m europarl_emi.etranslation.translate_unified prepare \
    --input data/translation/unified.parquet --id-col uid --text-col text --lang-col original_language \
    --date-col date --from-year 2013
sh scripts/etranslation.sh start
uv run python -m europarl_emi.score_unified          # EMI scores, ~15 min on an Apple-silicon GPU
```

Figures:

```bash
uv run python notebooks/paper/fig1.py        # -> results/figures/paper/fig1.pdf / .png
```

## Measure

Each speech is split into chunks of up to 150 words, embedded with the sentence-transformer `all-MiniLM-L6-v2`, and compared (cosine similarity) with the mean embedding of each lexicon. Similarities are adjusted for chunk length (10-word bins) and standardised over all chunks scored together; `EMI = z(evidence) − z(intuition)`, and the speech EMI is the mean over its chunks. Positive values mean more evidence-like language.

## Figures

**Figure 1.** (a) MEP oral speeches per year, delivered in English or translated with eTranslation. (b) Mean EMI per year with percentile bootstrap 95% intervals (B = 2000), for all speeches, speeches delivered in English, and translated speeches.

## References

- Aroyehun, S. T., Simchon, A., Carrella, F., Lasser, J., Lewandowsky, S. & Garcia, D. Computational analysis of US congressional speeches reveals a shift from evidence to intuition. *Nature Human Behaviour* **9**, 1122–1133 (2025). https://doi.org/10.1038/s41562-025-02136-2

## Contact

Jacopo Nudo
