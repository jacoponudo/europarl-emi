# europarl-emi

**Evidence versus intuition in European Parliament speeches, 1996–2023.**

This repository measures whether speeches in the European Parliament rely more on the language of *evidence* (data, research, proof, facts) or of *intuition* (belief, feeling, opinion, common sense), and how this balance has changed over time, across member states and across political families.

It extends to the European Parliament the approach of Aroyehun et al. (2025), who documented a shift from evidence-based to intuition-based language in US congressional speeches.

---

## Research question

> Does the language of European Parliament speeches shift from evidence towards intuition over time, and if so, is the shift driven by changes in *who* speaks (turnover of members, national and party composition) or by changes in *how* the same members speak?

## Method in brief

1. **Corpus.** English-language European Parliament speeches, 1996–2023.
2. **Procedural filtering.** Formulaic procedural texts ("The debate is closed", "Before the vote") are removed with a Naive Bayes–logistic regression (NB-LR) classifier trained on weak labels, following Card et al. (2022).
3. **Scoring.** Each speech is split into chunks of up to 150 words. Each chunk is embedded with the sentence-transformer `all-MiniLM-L6-v2`, together with two published keyword dictionaries: 49 evidence terms and 35 intuition terms, from Aroyehun et al. (2025). For every chunk we compute its cosine similarity to the average evidence vector and to the average intuition vector.
4. **EMI score.** Both similarities are adjusted for chunk length (10-word bins) and standardised **over the whole corpus**. The *evidence-minus-intuition* score is `EMI = z(evidence) − z(intuition)`. Speech-level EMI is the mean over a speech's chunks. Positive values mean more evidence-like language; negative values mean more intuition-like language.
5. **Main sample.** Until 2012 the Parliament published official English translations of speeches delivered in other languages, and the corpus also contains written statements and speeches by the chair, the Commission and the Council. To compare like with like over time, the main sample is restricted to **oral speeches by MEPs originally delivered in English**. These are identified from the speech header, for example `Name (GROUP). – (FR) …` marks a translation from French.
6. **Speakers.** Political group is taken from the speech header, so it reflects the group *at the time of the speech*, and harmonised into stable families (Left, Greens/EFA, S&D, Liberals, EPP, Conservatives, Eurosceptics, Non-attached). Country comes from the MEP identifier linked to the European Parliament's official membership data.
7. **Models.** Linear trends in EMI per decade, with speaker and/or agenda-item fixed effects and standard errors clustered by speaker.

## Pipeline

All steps live in `src/europarl_emi/` and are run as modules. Each step reads the previous step's output.

| Step | Module | What it does | Output |
|---|---|---|---|
| 1 | `step01_select_english` | Keep English-language speeches from the raw corpus | `data/interim/01_speeches_en.parquet` |
| 2 | `step02_clean_text` | Drop very short texts (< 3 words or < 16 characters), normalise whitespace (case and punctuation are kept) | `data/interim/02_speeches_clean.parquet` |
| 3 | `step03_filter_procedural` | Train the NB-LR procedural classifier (fixed seed) and remove procedural texts; save 200 examples for manual validation | `data/interim/03_speeches_substantive.parquet`, `results/procedural_check_sample.csv` |
| 4 | `step04_score_chunks` | Chunk speeches and compute similarities to the two lexicons with MiniLM (resumable) | `data/interim/04_chunk_scores/` |
| 5 | `step05_compute_emi` | Length adjustment, global standardisation, EMI per chunk and per speech | `data/processed/05_*.parquet` |
| 6 | `step06_parse_headers` | Parse speech headers: role, original language (translation), written statement, group | `data/processed/06_speech_annotated.parquet` |
| 7 | `step07_speakers` | Build the main sample; add country and political family | `data/processed/07_speech_main_sample.parquet` |
| 8 | `step08_models` | Trend models with fixed effects and clustered standard errors | `results/tables/trend_models.csv` |
| — | `figures` | All paper figures (Scientific Reports style) | `results/figures/paper/` |
| — | `inspect_data` | Inventory and sanity checks of input files | console |

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. uv installs Python if needed.

```bash
git clone https://github.com/jacoponudo/europarl-emi.git
cd europarl-emi
uv sync
```

Place the input data as described below, then run the pipeline:

```bash
uv run python -m europarl_emi.step01_select_english
uv run python -m europarl_emi.step02_clean_text
uv run python -m europarl_emi.step03_filter_procedural
uv run python -m europarl_emi.step04_score_chunks     # ~30 min on an Apple-silicon GPU (mps)
uv run python -m europarl_emi.step05_compute_emi
uv run python -m europarl_emi.step06_parse_headers
uv run python -m europarl_emi.step07_speakers
uv run python -m europarl_emi.step08_models
uv run python -m europarl_emi.figures                 # or: --only fig2 fig5
```

All paths and parameters (chunk sizes, thresholds, random seed, embedding model) are set in `src/europarl_emi/config.py`.

## Data

Data files are **not included** in this repository.

| Path | Content | Source |
|---|---|---|
| `data/raw/corpus/all_years_combined.csv.gzip` | European Parliament speeches, all languages, 1996–2023 (621,513 records) | Compiled corpus; provenance being documented |
| `data/raw/meps/meps_term{4..10}.csv` | MEP membership data by parliamentary term | [European Parliament Open Data Portal](https://data.europarl.europa.eu/en/datasets) (*Members of the European Parliament* datasets, terms 4–10) |
| `data/raw/lexicons/*.csv` | Evidence (49) and intuition (35) keyword dictionaries | Aroyehun et al. (2025); included in this repository |

Generated files go to `data/interim/` and `data/processed/`. The `models/` folder holds the trained procedural classifier.

## Repository structure

```
europarl-emi/
├── src/europarl_emi/      pipeline steps, models, figures, configuration
├── data/
│   ├── raw/               original inputs (never modified)
│   ├── interim/           intermediate outputs (regenerable)
│   └── processed/         analysis datasets
├── models/                trained classifier
├── notebooks/legacy/      original exploratory Colab notebooks (reference only)
└── results/
    ├── figures/paper/     paper figures (PDF + PNG)
    └── tables/            model estimates and supplementary tables
```

## Differences from the original exploratory notebooks

The pipeline replaces two Colab notebooks (`notebooks/legacy/`). The main changes:

- **Global standardisation.** The original code z-scored similarities within consecutive batches of 1,000 rows. Because the file was sorted by date, this forced the mean EMI of every year to zero and removed any time trend by construction (see Supplementary Fig. S9).
- **Speech-level aggregation.** Each speech counts once, instead of long speeches weighing more through their many chunks.
- **Like-for-like sample.** Translations, written statements and non-MEP speakers are separated from original English oral speeches.
- **Time-aware political groups.** Groups are read from each speech rather than from a single, first-listed affiliation per MEP; countries are linked by MEP identifier instead of fuzzy name matching.
- **Reproducibility.** Fixed random seeds, a scikit-learn classifier instead of TensorFlow/ktrain, and an explicit, resumable pipeline.
- **Unchanged.** The lexicons, the embedding model (`all-MiniLM-L6-v2`), the chunking rules and the procedural-filter heuristics are kept from the original work.

## Limitations

- The main sample covers speeches **originally delivered in English**, so UK and Irish MEPs are over-represented until 2020.
- Corpus coverage drops sharply after 2016; years after 2019 are reported but interpreted with caution.
- EMI measures semantic proximity to two keyword dictionaries, not the accuracy or quality of arguments.
- Aroyehun et al. (2025) used Word2Vec embeddings trained on the speeches themselves; this project uses a pretrained sentence-transformer.

## References

- Aroyehun, S. T., Simchon, A., Carrella, F., Lasser, J., Lewandowsky, S. & Garcia, D. Computational analysis of US congressional speeches reveals a shift from evidence to intuition. *Nature Human Behaviour* **9**, 1122–1133 (2025). https://doi.org/10.1038/s41562-025-02136-2
- Card, D. et al. Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration. *PNAS* **119** (2022).
- Lasser, J. et al. From alternative conceptions of honesty to alternative facts in communications by US politicians. *Nature Human Behaviour* **7** (2023).

## Contact

Jacopo Nudo
