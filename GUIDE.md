# Guide: data and code

This guide describes how the data are organised, what each file contains, and how the code produces them. The folders of `code/` mirror those of `data/`: `code/raw` writes `data/raw`, `code/translation` writes `data/translation`, `code/scores` writes `data/scores`, and `code/results` writes `results/`.

```
data/raw/inherited  ─┐
data/raw/api        ─┴─> data/translation ──> data/scores ──> results
   code/raw              code/translation      code/scores     code/results
```

Data never go into git (only the two lexicons do). They live on the Mac and on the JRC server, and move between machines through Nextcloud. Figures, tables and logs go to `results/`, which is not in git either.

## 1. Data

### 1.1 `data/raw/inherited/`: received as they are, never modified

| Path | Content |
|---|---|
| `corpus/all_years_combined.csv.gzip` | The inherited corpus of plenary speeches, all languages, 1996–2023: 621,513 records. Columns used: `guid` (id), `language`, `pubdate`, `description` (text, starting with the header), `detectedmepid`, `speaker`, `speaker-title`, `title` (agenda item). Until 2012 it holds the English version of the verbatim reports, translations included; from 2013 every speech is in the language in which it was delivered. |
| `meps/meps_term4.csv` … `meps_term10.csv` | Members of the European Parliament per parliamentary term (EP Open Data Portal). |
| `meps/mep_database_1994_present.csv` | MEP identifiers by term (`mep_id`, `term`): used to decide whether a speaker was a sitting MEP on the day of the speech. |
| `lexicons/evidence_lexicon.csv`, `intuition_lexicon.csv` | The 49 evidence and 35 intuition keywords of Aroyehun et al. (2025). The only data files in git. |
| `legacy_colab/` | Outputs of the original Colab notebooks (`outputv2*.csv.gzip`, `political_groups_simplified.csv`, `data_english_simplified.csv.gzip`, `dataenglishv2.csv.gzip`). Not regenerable and not used by the current code; kept for reference. |

### 1.2 `data/raw/api/`: downloaded from the Parliament's Open Data API

`speeches/YYYY-MM-DD.jsonl`, one file per weekday from 2021-07-01, one speech per line; days without a sitting give an empty file. The API has complete data only from July 2021. Fields:

| Field | Content |
|---|---|
| `speech_id` | API identifier of the speech |
| `date`, `start` | sitting day, start time |
| `activity_type` | `PLENARY_DEBATE_SPEECH` (oral), `PLENARY_DEBATE_WRITTEN_STATEMENT`, `MEETING_NAME_CHAIR`, `PROCEEDING_ACTIVITY` |
| `role` | `SPEAKER` or `CHAIR` |
| `person_id` | person code, the same id as in the MEP files |
| `agenda_item`, `agenda_title_en` | agenda item and its English title |
| `original_language` | language(s) in which the speech was delivered |
| `fragment_lang` | language of the text saved in `text` |
| `header` | speaker header, same form as in the corpus ("Name (GROUP)") |
| `text` | original-language text (the only human text) |
| `text_en_mt` | English machine translation provided by the API, for non-English speeches (often missing before 2024) |
| `xml` | original XML fragment |

### 1.3 `data/translation/`: one dataset for all speeches, and its translations

**`unified.parquet`**: one row per speech, 671,833 rows from 1996-04-15 to 2026-10-22: the corpus up to 30 June 2021 and the API from 1 July 2021, with the same columns for both and no speech counted twice. It was built on 2026-11-03, so API days downloaded later are not in it until `build_unified.py` is run again.

| Column | Content |
|---|---|
| `uid` | **speech id used everywhere**: `corpus:<guid>` or `api:<speech_id>` |
| `source`, `source_id` | `corpus` / `api` and the id in that source |
| `date`, `year`, `term` | sitting day, year, parliamentary term |
| `person_id`, `sitting_mep` | person id (digits only) and whether that person was an MEP in that term |
| `speaker`, `speaker_title` | corpus columns (empty for the API) |
| `agenda_title` | agenda item |
| `text_language`, `original_language` | language of `text`, language in which the speech was delivered (`unknown` if not known) |
| `header` | speaker header ("Name (GROUP)") |
| `text` | text as read by the measure: "header. – body" |
| `text_en_mt` | API machine translation, non-English API speeches only |
| `english_text` | where an English text is: `text (original)`, `text (human translation)` (corpus to 2012), `text_en_mt (machine translation)`, `none` |
| `role`, `written` | from the header rule (`headers.py`): mep / chair / commission / council / unknown; written statement |
| `activity_type`, `api_role` | API metadata |
| `category_header` | category from the header alone (the rule of the original paper) |
| `category` | homogeneous category over the two sources, used by the analyses: `MEP, oral, original English (main sample)`, `MEP, oral, translated`, `MEP, oral, other original language`, `MEP, oral, several languages`, `MEP, written statement`, `Chair, Commission or Council`, `Unrecognised header`, `Not English (outside Fig. 1)`, `Oral, speaker without person id`, `Other API activity (no Fig. 1 equivalent)` |
| `category_rule` | how the category was decided: `header`, `metadata`, `api` |
| `main_series` | `category` is the main sample |

**`etranslation.sqlite`**: translations into English, with the European Commission's eTranslation service, of every non-English text of `unified` from 2013 (before 2013 the corpus holds only the English version, so there is nothing to translate). `text_id` is the `uid` of `unified`. The translation is complete: 310,240 non-English texts, 310,232 fully translated, 312,667 segments all `done`.

| Table | Content |
|---|---|
| `texts` | one row per text: `text_id`, `hint_lang` (language in unified), `det_lang` and `det_conf` (language detected with lingua-py), `lang` (language sent to eTranslation), `lang_source` (`detected`, `hint`, `maltese_rule`, `short`), `n_chars`, `n_segments` |
| `segments` | texts split at sentence boundaries into segments of at most 4,500 characters: `text_id`, `idx` (order), `lang`, `text`, `status` (`new`/`sent`/`done`/`error`), `translation`, `request_id`, `sent_at`, `done_at`, `attempts`, `error` |
| `requests`, `meta` | eTranslation request ids and service state |

The English text of a translated speech is its `done` segments joined in `idx` order; `code/translation/unified.py` does this.

### 1.4 `data/scores/<model>_<window>_<date>/`: EMI scores

One folder per methodology, named model, window (words per chunk) and date of computation. Only scores computed on `unified` are kept. Currently there is one folder, `minilm_150_20261119`:

| File | Content |
|---|---|
| `speech_emi.parquet` | one row per speech: `uid`, `emi`, `evidence_z`, `intuition_z`, `n_chunks`, `n_words` |
| `meta.json` | model, window, population, texts scored, standardisation, counts, SD of speech EMI |
| `chunk_scores/` | raw similarities per chunk (`uid`, `chunk_id`, `n_words`, `sim_evidence`, `sim_intuition`) in parts, plus `manifest.json`: the slow part. Chunks already scored are reused, so adding speeches costs only their embeddings |

`minilm_150_20261119` covers the MEP oral speeches with an English text, in three groups: delivered in English, translated by hand by the Parliament (to 2012), and translated with eTranslation (from 2013). The counts are in `meta.json`. A first run on 19 November scored only the first and third groups (177,995 speeches: 63,472 delivered in English, 114,523 translated with eTranslation); the human translations (about 101,000) were then added to the same folder, and EMI was recomputed on all chunks together. Each folder is standardised on its own chunks, so values from different folders are not on the same scale.

### 1.5 `results/`

| Path | Content |
|---|---|
| `figures/paper/fig1.pdf`, `.png` | Figure 1 |
| `tables/fig1/` | `by_year.csv` (panel a), `emi_by_year.csv` (panel b: n, mean, sd, se and 95% interval per year and group) |
| `tables/unified/` | summaries written by `build_unified.py` |
| `logs/` | `build.log`, `etranslation/` (run, supervisor, tunnel, callbacks), `sensitivity/` (logs of the old variants) |
| `review/etranslation/` | earlier check of the EMI of translated speeches on the old paper scale (old code, history) |

## 2. Code

Every script runs from the project folder with `uv run python code/<folder>/<script>.py`. Each script puts `code/` on the Python path, so the folders import each other (`import config`, `from translation.unified import load`). All paths and parameters are in `code/config.py`.

### 2.1 `code/config.py`

Folders and files (`RAW_CORPUS`, `RAW_API`, `UNIFIED`, `ETRANSLATION_DB`, `SCORES`, `FIGURES`, `TABLES`, `LOGS`), seed (42), measure parameters (`all-MiniLM-L6-v2`, chunks of ≤ 150 words, last chunk < 50 words joined to the previous one, speeches of more than 10 words) and the start dates of the parliamentary terms.

### 2.2 `code/raw/`

**`download_speeches.py`** downloads every weekday from the API (`/api/v2/speeches`, 100 speeches per request, 1.2 s between requests; after HTTP 429 it waits 5 minutes) into `data/raw/api/speeches/`. Resumable: existing days are skipped. Days downloaded too soon after a sitting can come back empty: delete them and download again.

```bash
uv run python code/raw/download_speeches.py                          # 2021-07-01 to today
uv run python code/raw/download_speeches.py --start 2026-11-04
```

### 2.3 `code/translation/`

**`headers.py`**: `parse_header(text)` reads the header that opens every speech ("De Esteban Martin (PPE). - (ES) …") and returns header, original language if translated, written statement, role and political group.

**`build_unified.py`** builds `unified.parquet`: reads the corpus up to 30 June 2021 and the API from 1 July 2021, applies the header rule and the metadata rules, and assigns `category`. From 2020, when the corpus loses speaker names and groups, unrecognised headers are decided from the metadata. Summary tables go to `results/tables/unified/`.

```bash
mkdir -p results/logs && uv run python code/translation/build_unified.py 2>&1 | tee results/logs/build_unified.log
```

**`etranslation.py`** translates the non-English texts of `unified` with eTranslation REST v2, in four steps:

1. `prepare` detects the language of each text (lingua-py on the 24 EU languages, plus a rule for Maltese) and splits non-English texts into segments.
2. `run` sends the segments and receives translations through a callback on `127.0.0.1:8765`, made public with a cloudflared tunnel (works only from the Mac, not from the JRC network).
3. `status` counts segments by state and language.
4. `export` writes one English text per fully translated text.

Credentials are read only from the environment (`ETR_USER`, `ETR_PASS`) or from the Mac Keychain; never put them in a file.

**`etranslation.sh`** runs tunnel + translation in the background on the Mac and restarts them if they fail: `start`, `status`, `log`, `stop`, `restart`, `forget`.

```bash
uv run --with lingua-language-detector --with tqdm python code/translation/etranslation.py prepare \
    --input data/translation/unified.parquet --id-col uid --text-col text --lang-col original_language \
    --date-col date --from-year 2013
sh code/translation/etranslation.sh start
sh code/translation/etranslation.sh status
```

**`unified.py`**: `load(population="mep-oral", with_text=False)` returns `unified` with a `group` column (`original`, `human`, `etranslation`, `other`) and, with `with_text=True`, the English text of each speech in `en` (column `text` for `original` and `human`, the joined eTranslation segments for `etranslation`). Every analysis should start from here.

### 2.4 `code/scores/`

**`emi.py`** holds the measure, unchanged from the original pipeline: `split_into_chunks`, `lexicon_vectors` (mean embedding of each lexicon), `pick_device` (mps / cuda / cpu), `compute_emi`. `compute_emi` corrects similarities for chunk length (mean per 10-word bin), takes z-scores over all chunks given, and returns EMI = z(evidence) − z(intuition).

**`score_unified.py`** scores the English texts of `unified` (delivered in English, translated by hand by the Parliament, or translated with eTranslation; header included) and writes `data/scores/minilm_150_<date>/`. Embeddings are saved in parts and reused by `(uid, chunk_id)`: an interrupted run resumes, a run that adds speeches embeds only the new chunks, and EMI is always recomputed on all chunks. If a text changes, its chunks no longer match the cache and the script stops: use a new `--date`.

```bash
uv run python code/scores/score_unified.py                    # folder minilm_150_<today>, ~15 min on the Mac GPU
uv run python code/scores/score_unified.py --date 20261119    # reuse that folder
```

### 2.5 `code/results/`

**`style.py`**: figure style (Nature Human Behaviour: Arial 7 pt, bold lowercase panel letters, widths 89/120/183 mm, editable PDF text) and the project palette (`PALETTE`, nine colours from Cherry Rose `#b7094c` to Pacific Cyan `#0091ad`).

**`fig1.py`**: Figure 1, MEP oral speeches in three groups: delivered in English, human translation by the Parliament (to 2012), translated with eTranslation (from 2013).
- Panel a: speeches per year in each group.
- Panel b: mean EMI per year with 95% confidence intervals computed from the data, mean ± t(0.975, n−1) · sd/√n with one speech as one observation, for each group and for all speeches together (dashed dark grey: the palette has no fourth hue that stays distinct).
- `--without-human` leaves the human translations out of both panels and of "all speeches".

```bash
uv run python code/results/fig1.py
```

### 2.6 Full run, in order

```bash
uv sync
uv run python code/raw/download_speeches.py
uv run python code/translation/build_unified.py
uv run --with lingua-language-detector --with tqdm python code/translation/etranslation.py prepare ...   # see 2.3
sh code/translation/etranslation.sh start
uv run python code/scores/score_unified.py
uv run python code/results/fig1.py
```

### 2.7 Adding things

- **A new figure**: `code/results/figN.py`. Start from `translation.unified.load`, read scores from `data/scores/<folder>/speech_emi.parquet` merged on `uid`, use `results.style`, and write to `results/figures/paper/` and `results/tables/figN/`.
- **A new methodology**: a script in `code/scores/` that writes `data/scores/<model>_<window>_<date>/speech_emi.parquet` (`uid`, `emi`) and `meta.json`.
- **Before every commit**: check that no data go into git.

```bash
git diff --cached --name-only | grep -E '^(data|results|paper|models)/' && echo STOP || echo ok
```

## 3. Known issues

- **Coverage 2020 – June 2021.** The corpus is incomplete in 2020 and the first half of 2021, and the API starts only in July 2021. Figure 1 shades these years.
- **Possible written statements among the translated oral speeches, 2015–2016.** These years have 12,695 and 14,078 translated MEP oral speeches, against about 7,500–8,800 in other years. They also coincide with the peak of written statements in English. In those years the translated speeches have a mean EMI 0.2–0.4 higher than speeches delivered in English. To check: count translated speeches whose translated header contains "in writing"; if they are many, correct the rule in `build_unified.py`.
- **`unified` is from 2026-11-03.** API days downloaded afterwards need a new run of `build_unified.py`, then of `etranslation.py prepare/run` for the new non-English texts, and of `score_unified.py` with a new date.
- **Two translation methods.** The translated speeches are translated by hand by the Parliament up to 2012 and by eTranslation from 2013; the originals of the human-translated speeches are not in the corpus, so the two methods cannot yet be compared on the same speeches.
- **Small gaps.** 8 non-English texts are not fully translated; speeches of 10 words or fewer get no EMI.
