# Fragrance Explorer

A local Streamlit app for navigating the world of perfumery using a personal taste profile built from your own ratings. The interesting part isn't the recommendation engine per se — it's the question of whether a weighted TF-IDF vector over olfactory note pyramids can actually capture something real about taste, and whether an LLM can say something useful about *why* a fragrance matches.

This was built to explore what LLMs and classic NLP can do in a domain I care about, starting from a ~20k fragrance dataset scraped from Fragrantica.

<!-- SCREENSHOT: full app overview showing all seven tabs -->

---

## How the algorithm works

### Representing fragrances as vectors

Each fragrance in the dataset is a bag of notes: top, middle, base, and main accords concatenated into a single string. A TF-IDF vectorizer fitted on the full corpus assigns weights based on note rarity — "musk" and "cedar" score lower than "castoreum" or "petitgrain," because they distinguish less. The tokenizer preserves comma-separated note lists as individual tokens, so "Bergamot, Lemon" becomes two features, not one.

### The personal profile vector

`build_personal_profile` in [recommender.py](recommender.py) constructs a synthetic document from your rated fragrances. Each fragrance's note string is repeated 1–3 times, proportional to a min-max normalized weight derived from your rating. The concatenated result, passed through the same TF-IDF vectorizer as the dataset, lands in the same high-dimensional space as every catalogue entry. Cosine similarity then finds the nearest neighbors.

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

The app has seven tabs.

**La mia lista** is your personal catalogue. Each entry expands to show olfactory notes, ownership status (Decant, Full Bottle, Sample, Wishlist), bottle candidate flag, and personal comments. You can filter by minimum rating, toggle to show only bottle candidates, and sort by rating or brand. Each entry has an inline edit form covering rating, comment, ownership, and bottle candidate, plus a delete button with a two-step confirmation.

<!-- SCREENSHOT: La mia lista tab with an entry expanded -->

**Abbina note** links your personal fragrances to the Kaggle dataset so they can participate in recommendations. A two-stage fuzzy match (rapidfuzz `fuzz.ratio` on brand, then `fuzz.token_sort_ratio` on name) surfaces up to 15 candidates per fragrance, each showing the full note pyramid and accords. If the automatic candidates miss, there's a manual dataset search by name or brand. Confirmed matches are written to `data/confirmed_matches.csv` and the note data is back-propagated to your personal CSV. The tab shows running stats — total, confirmed, skipped, pending — so you can see at a glance how much of your list is active in the recommendation engine.

<!-- SCREENSHOT: Abbina note tab showing candidate selection for a fragrance -->

**Profilo olfattivo** visualizes your taste. Six summary metrics span the top (total fragrances, average rating, full bottles, decants, bottle candidates, dominant olfactory family). Below them, a Plotly horizontal bar chart shows rating distribution. Beside it, a radar chart maps your top 8 olfactory families, weighted by rating — accord keywords are matched against each fragrance's combined note string, and the score is a sum of rating-normalized weights, normalized to a 0–1 scale. The scatter plot of your full collection anchors the tab. A button at the bottom triggers an LLM narrative profile: Claude or a local Ollama model reads your top-rated and low-rated fragrances and writes a 150–200 word assessment.

<!-- SCREENSHOT: Profilo olfattivo tab showing stats, radar chart, and scatter plot -->

**Raccomandazioni** is the main recommendation engine. Two modes: profile-based (uses the full personal vector) and "similar to a specific fragrance" (builds a temporary single-fragrance profile with weight 1.0). Controls include recommendation count, gender filter, an option to exclude already-known fragrances, and the quality slider. Each result shows notes and accords in an expander. Clicking "Spiega perché" fires the LLM with the candidate's full note data, your top-rated fragrances by name and rating, and your profile's top note cluster — asking it to explain the connection by referencing your actual favorites, not generic hedges.

<!-- SCREENSHOT: Raccomandazioni tab with a recommendation expanded and LLM explanation visible -->

**Esplora stili** is for venturing outside your comfort zone. Pick one of 25 named styles, set the intensity, apply gender and quality filters, and the engine returns fragrances that match the style vocabulary while still respecting your avoid list. An LLM explanation runs automatically on each result set, describing why the returned fragrances are good entry points into that style for someone with your specific taste history.

**Aggiungi** is the manual entry form: brand, name, format (EdP/EdT/Extrait/Parfum/EdC), ownership, rating on a 1.0–10.0 half-step slider, bottle candidate checkbox, and a comment field. Duplicate detection runs on submit — if the brand and name already exist, a warning appears with an option to force-add anyway (useful for multiple formats of the same fragrance).

**Arricchisci note** handles fragrances that didn't match in the dataset or matched but had no pyramid data. A DuckDuckGo search finds the Fragrantica URL; the page is scraped with BeautifulSoup, parsing `pyramid-note-label` spans and the `rounded-br-lg` accord container. If the auto-found URL is wrong, you can edit it in place and re-scrape. Confirmed notes write to the ratings CSV with a `notes_source: web_enriched` marker.

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
pip install streamlit pandas scikit-learn anthropic beautifulsoup4 \
            plotly rapidfuzz openai duckduckgo-search
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
├── app.py          — Streamlit UI: seven tabs, all interaction logic
├── recommender.py  — TF-IDF, cosine similarity, profile building, scatter data
├── matcher.py      — dataset loading, fuzzy matching, note enrichment from dataset
├── enricher.py     — DuckDuckGo search, Fragrantica scraping
├── parser.py       — HTML import, CSV persistence, add/edit/delete ratings
├── llm.py          — LLM backend (Claude or Ollama), three prompt functions
├── config.py       — ACCORDS list, EXPLORATION_STYLES dict, rating constants
├── launch.bat      — Windows one-click launcher
└── data/
    ├── my_ratings.csv          — personal fragrance database
    └── confirmed_matches.csv   — dataset link table (personal ↔ Kaggle index)
```

---

## Roadmap

The three items below were in the original roadmap. All have since shipped.

- [x] Scatter plot of personal collection (freshness × depth, sized by intensity, colored by rating)
- [x] Enrich notes tab — web lookup for fragrances missing from the dataset
- [x] Delete entry from personal list with confirmation

What might come next: export of the personal profile as a shareable JSON, a diff view when re-matching a fragrance after notes change, and a smarter deduplication pass across the two source datasets.

---

## License

MIT — see [LICENSE](LICENSE)
