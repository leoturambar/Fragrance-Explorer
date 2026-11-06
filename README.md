# Fragrance Explorer

A local Streamlit app for navigating the world of perfumery using a personal taste profile built from your own ratings. The interesting part isn't the recommendation engine per se — it's the question of whether a weighted TF-IDF vector over olfactory note pyramids can actually capture something real about taste, and whether an LLM can say something useful about *why* a fragrance matches.

This was built to explore what LLMs and classic NLP can do in a domain I care about, starting from a ~20k fragrance dataset scraped from Fragrantica.

<!-- SCREENSHOT: full app overview showing all four tabs -->

---

## How the algorithm works

### Representing fragrances as vectors

Each fragrance in the dataset is a bag of notes: top, middle, base, and main accords concatenated into a single string. A TF-IDF vectorizer fitted on the full corpus assigns weights based on note rarity — "musk" and "cedar" score lower than "castoreum" or "petitgrain," because they distinguish less. The tokenizer preserves comma-separated note lists as individual tokens, so "Bergamot, Lemon" becomes two features, not one.

### The personal profile vector

`build_personal_profile` in [recommender.py](recommender.py) constructs a synthetic document from your rated fragrances. Each fragrance's note string is repeated 1–3 times, proportional to a combined weight: a min-max normalized rating weight multiplied by a status multiplier based on how you own the fragrance.

**Status multipliers:**

| Status | Bottle Candidate | Weight |
|---|---|---|
| Bottle | — | 1.5× |
| Decant / Sample / Tested | Yes | 1.2× |
| Decant / Sample / Tested | No | 1.0× |
| No status, Watchlist only | — | excluded from profile |

Fragrances you own bottles of pull the profile harder than samples; fragrances you've only wishlisted don't contribute at all. The Watchlist tag is a standalone bookmark — orthogonal to status — and can be set regardless of whether you own the fragrance.

The profile tracks two additional dimensions beyond the main vector. **Top notes** are extracted from fragrances rated ≥ 8.0 and used as context when the LLM explains recommendations. **Avoid notes** come from fragrances rated ≤ 5.5 and become a penalty: similarity to the avoid vector is subtracted at 30% weight from every candidate's score. This means fragrances that smell like things you've already rated poorly get pushed down the list even if they share notes with things you love.

The profile drifts as you rate more fragrances. It doesn't average — it accumulates in proportion to intensity of preference.

### Quality scoring

High similarity to your profile doesn't mean a fragrance is worth recommending. A quality score combines individual reputation (Fragrantica community rating × log1p(review count), normalized) with brand-level reputation (average rating × log1p(median reviews per fragrance) across the brand's catalogue, normalized). The combination is 70% individual + 30% brand. A slider in the UI controls a quality threshold and multiplicative penalty, letting you trade curation for breadth.

### Exploration mode

Instead of the personal profile as the query, exploration mode uses a fixed vocabulary string for a target style — for example, the Oud query is `"oud agarwood resinous smoky incense woody barnyard"`. There are 25 named styles defined in [config.py](config.py), spanning everything from Aquatic and Fougère to Tobacco and Animalic. An intensity slider (0–1) blends style similarity against profile similarity: at 0, results must still be close to your profile; at 1, only style relevance counts. The avoid penalty applies regardless of intensity.

### Scatter plot

`compute_scatter_data` in [recommender.py](recommender.py) places each matched fragrance in a 2D freshness × depth space. Note keyword sets — citrus, aquatic, herbal for freshness; oud, amber, tobacco for depth — are scored against each pyramid level with weights of 0.3 (top), 0.6 (middle), and 1.0 (base), reflecting the fact that base notes are what actually persists. Accord labels provide additional multiplier boosts. Intensity (marker size) is computed from a separate "heavy" note set. All three dimensions go through log1p before max-normalization to compress the long tail. Jitter is added at render time so overlapping points separate visually.

<!-- SCREENSHOT: scatter plot showing personal collection in freshness × depth space, sized by intensity, colored by rating -->

---

## Features

The app has four tabs.

**La mia lista** is your personal catalogue. A search box looks through brand, name, notes and comments; filters cover a rating range, status (Bottle / Decant / Sample / Tested / Watchlist / no status), bottle candidates and entries without a note pyramid; sorting by rating, brand or date added. Two views: *Schede* (expandable cards showing notes, where they came from — dataset match or Fragrantica — and an inline edit form for every field including brand, name and format) and *Tabella* (an editable grid for bulk changes, saved with one click). Delete requires a two-step confirmation. Every entry has a stable id, so deleting or reordering never disturbs other entries.

The ownership model has three independent fields. **Status** (Bottle / Decant / Sample / Tested) is mutually exclusive and tracks how you possess or have tried the fragrance. **Bottle Candidate** is a boolean that flags fragrances you're considering buying; it's hidden when status is already Bottle. **Watchlist** is a freestanding bookmark, available at any status level including none — useful for tracking fragrances you want to try without claiming you've tried them.

<!-- SCREENSHOT: La mia lista tab with an entry expanded -->

**Profilo** visualizes your taste, using only fragrances you have actually tried (watchlist-only bookmarks are excluded). Summary metrics span the top. A rating-distribution bar chart is colored by liked / neutral / disliked. The radar compares olfactory families among liked (≥ 7.5) and disliked (≤ 6) fragrances, matching families as whole words so 'rose' doesn't count 'rosewood'. The scatter plot (freshness × depth, sized by intensity, colored by rating relative to your own range, shaped by status) has its own visibility filter, optional name labels, and click-to-inspect. An LLM narrative profile (button at the bottom) writes a 150–200 word assessment.

**Gap analysis** compares how common each family is in your list vs the whole dataset, highlighting under-explored territory. **Deriva temporale** plots ratings over time; it needs the `date_added` field, which is recorded for every fragrance added from now on.

<!-- SCREENSHOT: Profilo tab showing stats, radar chart, and scatter plot -->

**Scopri** has three modes behind one toggle — *Dal mio profilo*, *Simile a…*, *Esplora uno stile* — sharing the count, gender (Maschile + Unisex by default, matching the dataset's "for men" / "for women" / "for women and men" labels) and quality controls. Fragrances already in your list are excluded via confirmed matches plus fuzzy brand+name matching that tolerates spelling differences ("Memo" vs "Memo Paris", catalogue numbers like "02 L'Air du Desert Marocain"). Any result can be bookmarked to the watchlist with one click, notes included.

*Dal mio profilo* uses the full personal vector; *Simile a…* builds a single-fragrance profile from one entry of your list. Clicking "Spiega perché" asks the LLM to explain the connection by referencing your actual favorites. Results and explanations are kept in session state so they survive any interaction.

*Esplora uno stile* is for venturing outside your comfort zone. Pick one of 25 named styles and set the intensity slider (0 = close to your profile, 1 = maximum style departure). The LLM explanation runs only when you ask for it, so changing other widgets never re-triggers it.

<!-- SCREENSHOT: Scopri tab with recommendations expanded -->

**Gestione** consolidates the data-management tools in inner tabs.

*Aggiungi* is the manual entry form: brand, name, format, status, Watchlist, Bottle Candidate, rating (or "not tried yet" for a pure bookmark) and comment. Duplicate detection considers brand, name and format. Right after adding, the best dataset candidates are offered so you can attach notes in one click.

*Abbina note* links your fragrances to the dataset. You pick one entry at a time and get up to 10 fuzzy candidates (brand prefix/word matching, name compared after stripping brand, concentration and edition words), each with its pyramid and accords. The match is stored on the entry itself as `ds_brand` / `ds_name` (never as a row position). If the dataset row has only accords, existing notes (e.g. from Fragrantica) are kept. A manual dataset search is also available.

*Arricchisci note* completes entries without a note pyramid: a DuckDuckGo + Fragrantica lookup (Fragrantica often answers 403 to automated requests) and a manual form to paste or correct top / heart / base notes and accords.

*Salute dati* runs automatic checks — matches whose name doesn't resemble the entry, references missing from the dataset, stored notes that differ from the matched row, entries without notes — and offers a CSV backup download.

---

## Data

The app merges two local CSV files. `data/fragrances_clean.csv` is a semicolon-delimited dataset with pre-parsed note columns. `data/fragrances_raw.csv` has URL-based entries where brand and name are extracted from the URL path and notes are parsed out of a free-text description field via regex. Deduplication is done by normalizing brand + name (lowercase, unicode normalization, stripping dashes and spaces) and keeping the clean version on conflicts.

Neither CSV is included in the repository. The clean dataset corresponds to a Fragrantica community scrape available on Kaggle. You can also bootstrap from a personal HTML export by placing it at `data/my_list.html` — the parser reads sections with `.score` elements and items with `.brand`, `.name`, `.form`, and `.bc` spans.

---

## LLM backend

The LLM layer is dual-backend, toggled by `LLM_BACKEND` in [llm.py](llm.py). Set to `"claude"`, it calls the model in `CLAUDE_MODEL` via the Anthropic API. Set to `"ollama"`, it calls a local Ollama server (`OLLAMA_MODEL`, currently `qwen3.5:9b`). Either way, the three prompt functions — `explain_recommendation`, `explain_profile`, `explain_exploration` — pass the same structured context: rated fragrance names, ratings, comments, and extracted profile data.

---

## Setup

Python 3.10+ is required. Install dependencies:

```bash
pip install -r requirements.txt
```

Place your datasets at `data/fragrances_clean.csv` and `data/fragrances_raw.csv`. Set your API key if using Claude:

```bash
# Linux/macOS
export ANTHROPIC_API_KEY=your_key_here

# Windows (persists across sessions)
setx ANTHROPIC_API_KEY your_key_here
```

Then run:

```bash
streamlit run app.py
```

On Windows, double-clicking `launch.bat` activates the `fragrance` conda environment and starts the app — no terminal needed.

---

## Project structure

```
fragrance-explorer/
├── app.py              # Streamlit UI: four tabs (La mia lista, Profilo, Scopri, Gestione)
├── recommender.py      # TF-IDF, cosine similarity, profile building, scatter data
├── matcher.py          # dataset loading, fuzzy matching, note enrichment from dataset
├── enricher.py         # DuckDuckGo search, Fragrantica scraping
├── ratings.py          # HTML import, CSV persistence, add/edit/delete ratings, migration
├── llm.py              # LLM backend (Claude or Ollama), three prompt functions
├── config.py           # ACCORDS list, EXPLORATION_STYLES dict, rating constants
├── launch.bat          # Windows one-click launcher
├── data/               # Input database and preferences
└── assets/             # Screenshots
```

---

## Roadmap

Items that have shipped:

- [x] Scatter plot of personal collection (freshness × depth, sized by intensity, colored by rating)
- [x] Enrich notes tab — web lookup for fragrances missing from the dataset
- [x] Delete entry from personal list with confirmation
- [x] Status/Watchlist ownership model with per-status profile weights
- [x] Data integrity overhaul — stable ids, matches stored on each entry, health checks, gender filter and known-fragrance exclusion fixed
- [x] Consolidated 4-tab layout (was 7)
- [x] Gap analysis, liked-vs-disliked radar, editable table view, search, watchlist from recommendations

What might come next: Deriva temporale with enough dated entries, export of the personal profile as a shareable JSON.

---

## License

MIT — see [LICENSE](LICENSE)
