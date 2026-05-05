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

**La mia lista** is your personal catalogue. A graph visibility filter at the top — Tutti / Provati o posseduti / Solo posseduti / Solo bottiglie — is stored in session state and shared with the Profilo scatter plot. The list can be filtered by minimum rating, status (Bottle / Decant / Sample / Tested / Watchlist / no status), and bottle candidate flag, then sorted by rating or brand. Each entry expands to show olfactory notes, status, Watchlist flag, and personal comments. The inline edit form covers rating, comment, status, Watchlist, and Bottle Candidate. Delete requires a two-step confirmation.

The ownership model has three independent fields. **Status** (Bottle / Decant / Sample / Tested) is mutually exclusive and tracks how you possess or have tried the fragrance. **Bottle Candidate** is a boolean that flags fragrances you're considering buying; it's hidden when status is already Bottle. **Watchlist** is a freestanding bookmark, available at any status level including none — useful for tracking fragrances you want to try without claiming you've tried them.

<!-- SCREENSHOT: La mia lista tab with an entry expanded -->

**Profilo** visualizes your taste. Six summary metrics span the top: total fragrances, average rating, bottle count, decant count, bottle candidates, and dominant olfactory family. Below them, a Plotly horizontal bar chart shows rating distribution. Beside it, a radar chart maps your top 8 olfactory families, weighted by rating — accord keywords are matched against each fragrance's combined note string, and the score is a sum of rating-normalized weights, normalized to a 0–1 scale. The scatter plot of your collection uses the graph visibility filter set in La mia lista. An LLM narrative profile (button at the bottom) asks Claude to read your top-rated and low-rated fragrances and write a 150–200 word assessment.

Two placeholder expanders mark future sections: **Deriva temporale** (taste drift over time) and **Gap analysis** (unexplored olfactory families).

<!-- SCREENSHOT: Profilo tab showing stats, radar chart, and scatter plot -->

**Scopri** merges the two discovery modes behind a radio toggle.

*Basato sul tuo profilo* is the main recommendation engine. Two sub-modes: profile-based (uses the full personal vector) and "similar to a specific fragrance" (builds a temporary single-fragrance profile with weight 1.0). Controls include recommendation count, gender filter, an option to exclude already-known fragrances, and the quality slider. Each result shows notes and accords in an expander. Clicking "Spiega perché" fires the LLM with the candidate's full note data, your top-rated fragrances, and your profile's top note cluster — asking it to explain the connection by referencing your actual favorites. Recommendations and individual explanations are stored in session state so they survive any UI interaction without recomputing.

*Esplora uno stile* is for venturing outside your comfort zone. Pick one of 25 named styles, set the intensity slider (0 = close to your profile, 1 = maximum style departure), apply gender and quality filters, and the engine returns fragrances matching the style vocabulary while respecting your avoid list. An LLM explanation runs automatically on each result set.

<!-- SCREENSHOT: Scopri tab with recommendations expanded -->

**Gestione** consolidates three data-management tools in inner tabs.

*Aggiungi* is the manual entry form: brand, name, format (EdP/EdT/Extrait/Parfum/EdC), status, Watchlist checkbox, Bottle Candidate checkbox, rating on a 1.0–10.0 half-step slider, and a comment field. Duplicate detection runs on submit — if brand and name already exist, a confirmation dialog appears outside the form (so its buttons survive Streamlit's rerun cycle) with an option to force-add anyway, useful for multiple formats of the same fragrance.

*Abbina note* links your personal fragrances to the dataset so they can participate in the recommendation engine. A two-stage fuzzy match (rapidfuzz `fuzz.ratio` on brand, then `fuzz.token_sort_ratio` on name) surfaces up to 15 candidates per fragrance, each showing the full note pyramid and accords. If automatic candidates miss, a manual dataset search by name or brand is available. Confirmed matches are stored in `data/confirmed_matches.csv` with brand and name as the stable lookup key (not just an integer index) and back-propagated to the ratings CSV. The sub-tab shows running stats — total, confirmed, skipped, pending.

*Arricchisci note* handles fragrances that didn't match in the dataset or matched but had no pyramid data. A DuckDuckGo search finds the Fragrantica URL; the page is scraped with BeautifulSoup, parsing `pyramid-note-label` spans and the `rounded-br-lg` accord container. If the auto-found URL is wrong, you can edit it in place and re-scrape. Confirmed notes write to the ratings CSV with a `notes_source: web_enriched` marker.

---

## Data

The app merges two local CSV files. `data/fragrances_clean.csv` is a semicolon-delimited dataset with pre-parsed note columns. `data/fragrances_raw.csv` has URL-based entries where brand and name are extracted from the URL path and notes are parsed out of a free-text description field via regex. Deduplication is done by normalizing brand + name (lowercase, unicode normalization, stripping dashes and spaces) and keeping the clean version on conflicts.

Neither CSV is included in the repository. The clean dataset corresponds to a Fragrantica community scrape available on Kaggle. You can also bootstrap from a personal HTML export by placing it at `data/my_list.html` — the parser reads sections with `.score` elements and items with `.brand`, `.name`, `.form`, and `.bc` spans.

---

## LLM backend

The LLM layer is dual-backend, toggled by `LLM_BACKEND` in [llm.py](llm.py). Set to `"claude"`, it calls `claude-haiku-4-5-20251001` via the Anthropic API. Set to `"ollama"`, it hits `http://localhost:11434/v1` with `qwen2.5:14b` using the OpenAI-compatible endpoint. Either way, the three prompt functions — `explain_recommendation`, `explain_profile`, `explain_exploration` — pass the same structured context: rated fragrance names, ratings, comments, and extracted profile data.

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
- [x] Graph visibility filter shared across La mia lista and Profilo
- [x] Consolidated 4-tab layout (was 7)
- [x] Stable dataset matching — confirmed matches stored by brand+name, not integer index

What might come next: Deriva temporale (taste drift over time), Gap analysis (unexplored olfactory families), export of the personal profile as a shareable JSON.

---

## License

MIT — see [LICENSE](LICENSE)
